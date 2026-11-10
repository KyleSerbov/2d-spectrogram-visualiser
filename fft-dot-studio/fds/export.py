"""Offline, frame-exact video export through ffmpeg.

Frame k is rendered at t = k / fps (the project's analysis alignment decides which audio a frame shows;
preview output latency never applies). The timeline audio is
mixed into a 24-bit WAV padded to exactly n_frames / fps seconds, then muxed (AAC 320k, FLAC or PCM)
so audio and video both start at 0 and have the same length."""
import os
import queue
import subprocess
import tempfile
import threading
import time
from functools import lru_cache

import numpy as np

from .audio_io import find_ffmpeg

SR = 44100
RESOLUTIONS = {'1280x720 (720p)': (1280, 720), '1920x1080 (1080p)': (1920, 1080), '2560x1440 (1440p)': (2560, 1440),
               '2880x1620': (2880, 1620), '3840x2160 (4K)': (3840, 2160), '1080x1080 (square)': (1080, 1080),
               '1080x1920 (vertical)': (1080, 1920)}
FPS_CHOICES = [24, 25, 30, 50, 60]
CODECS = ['auto', 'h264_nvenc', 'hevc_nvenc', 'libx264', 'libx265']
AUDIO_CODECS = ['aac', 'flac', 'pcm']
_NOWIN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def ffprobe_path():
    ff = find_ffmpeg()
    if not ff:
        return None
    cand = os.path.join(os.path.dirname(ff), 'ffprobe.exe' if os.name == 'nt' else 'ffprobe')
    return cand if os.path.exists(cand) else 'ffprobe'


def available_encoders():
    ff = find_ffmpeg()
    if not ff:
        return set()
    out = subprocess.run([ff, '-hide_banner', '-encoders'], capture_output=True, text=True, creationflags=_NOWIN).stdout
    return {line.split()[1] for line in out.splitlines() if len(line.split()) > 1 and line.startswith(' V')}


@lru_cache(maxsize=None)
def nvenc_works(codec='h264_nvenc'):
    ff = find_ffmpeg()
    if not ff:
        return False
    r = subprocess.run([ff, '-v', 'error', '-f', 'lavfi', '-i', 'color=black:s=256x256:d=0.1', '-c:v', codec,
                        '-f', 'null', '-'], capture_output=True, creationflags=_NOWIN)
    return r.returncode == 0


class Exporter:
    def __init__(self, engine, gl_ctx, path, width=1920, height=1080, fps=30, codec='auto', audio='aac', quality=18,
                 t_start=0.0, t_end=None):
        from .renderer import Renderer
        self.engine = engine
        self.path = path
        self.w, self.h = int(width) // 2 * 2, int(height) // 2 * 2
        self.fps = int(fps)
        self.t_start = t_start
        total = engine.total_duration() if t_end is None else t_end
        self.duration = max(0.0, total - t_start)
        self.n_frames = int(np.ceil(self.duration * self.fps - 1e-6))
        self.exact_duration = self.n_frames / self.fps
        self.k = 0
        self.error = None
        self.done = False
        self.cancelled = False
        self.proc = None
        self.started = time.perf_counter()
        self.renderer = Renderer(gl_ctx, self.w, self.h)
        self.codec = codec
        self.audio = audio
        self.quality = quality
        self._tmpdir = tempfile.mkdtemp(prefix='fds_export_')
        self.wav = os.path.join(self._tmpdir, 'audio.wav')

    def _video_args(self):
        codec = self.codec
        if codec == 'auto':
            codec = 'h264_nvenc' if nvenc_works('h264_nvenc') else 'libx264'
        q = str(int(self.quality))
        if codec in ('h264_nvenc', 'hevc_nvenc'):
            return ['-c:v', codec, '-preset', 'p5', '-rc', 'vbr', '-cq', q, '-b:v', '0', '-pix_fmt', 'yuv420p'], codec
        if codec == 'libx265':
            return ['-c:v', 'libx265', '-crf', q, '-preset', 'medium', '-pix_fmt', 'yuv420p', '-tag:v', 'hvc1'], codec
        return ['-c:v', 'libx264', '-crf', q, '-preset', 'medium', '-pix_fmt', 'yuv420p'], codec

    def _audio_args(self):
        ext = os.path.splitext(self.path)[1].lower()
        if self.audio == 'pcm':
            if ext not in ('.mov', '.mkv'):
                raise RuntimeError('PCM audio needs a .mov or .mkv output file')
            return ['-c:a', 'pcm_s24le']
        if self.audio == 'flac':
            return ['-c:a', 'flac', '-strict', '-2'] if ext == '.mp4' else ['-c:a', 'flac']
        return ['-c:a', 'aac', '-b:a', '320k']

    def start(self):
        ff = find_ffmpeg()
        if not ff:
            raise RuntimeError('ffmpeg not found (install with: winget install Gyan.FFmpeg)')
        if self.n_frames <= 0:
            raise RuntimeError('Nothing to export (timeline is empty)')
        self._write_audio()
        vargs, self.codec_used = self._video_args()
        cmd = [ff, '-y', '-v', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{self.w}x{self.h}', '-framerate', str(self.fps), '-i', '-',
               '-i', self.wav, '-map', '0:v:0', '-map', '1:a:0', *vargs, *self._audio_args(),
               '-r', str(self.fps), '-t', f'{self.exact_duration:.6f}']
        if self.path.lower().endswith(('.mp4', '.mov')):
            cmd += ['-movflags', '+faststart']
        cmd.append(self.path)
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NOWIN)
        self._stderr = []
        self._write_err = None
        self._frames = queue.Queue(maxsize=8)
        self._closing = False
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._writer = threading.Thread(target=self._write_frames, daemon=True)
        self._writer.start()

    def _drain_stderr(self):
        for line in self.proc.stderr:
            self._stderr.append(line.decode(errors='replace'))

    def _write_frames(self):
        # Pipe writes block while ffmpeg encodes; doing them here keeps the UI thread free.
        while True:
            buf = self._frames.get()
            if buf is None:
                break
            try:
                self.proc.stdin.write(buf)
            except (BrokenPipeError, OSError) as e:
                self._write_err = e
                break
        try:
            self.proc.stdin.close()
        except OSError:
            pass

    def _ffmpeg_err(self):
        return ''.join(self._stderr).strip()[-500:]

    def _write_audio(self):
        import soundfile as sf
        n_total = int(round(self.exact_duration * SR))
        start = int(round(self.t_start * SR))
        offs, _ = self.engine.offsets()
        segs = [(int(round(o * SR)) - start, ta.audio) for o, ta in zip(offs, self.engine.tracks()) if ta.audio is not None]
        chunk = SR * 10
        with sf.SoundFile(self.wav, 'w', SR, 2, subtype='PCM_24', format='WAV') as f:
            for pos in range(0, n_total, chunk):
                n = min(chunk, n_total - pos)
                out = np.zeros((n, 2), np.float32)
                for s0, arr in segs:
                    a, b = max(pos, s0), min(pos + n, s0 + len(arr))
                    if b > a:
                        out[a - pos:b - pos] += arr[a - s0:b - s0].astype(np.float32) / 32767.0
                f.write(out)

    def frame_time(self, k):
        return self.t_start + k / self.fps

    def step(self, budget_s=0.05):
        """Render frames for up to budget_s seconds. Returns True while still running."""
        if self.done or self.cancelled or self.error:
            return False
        if self._write_err is not None or (self.proc.poll() not in (None, 0)) or (
                self.proc.poll() is not None and self.k < self.n_frames):
            self.proc.wait()
            self.error = f'ffmpeg failed: {self._ffmpeg_err() or self._write_err}'
            self._cleanup()
            return False
        t_end = time.perf_counter() + budget_s
        aspect = self.w / self.h
        while self.k < self.n_frames and time.perf_counter() < t_end:
            if self._frames.full():
                break
            dots = self.engine.gather(self.frame_time(self.k), aspect)
            self.renderer.render(dots)
            self._frames.put(self.renderer.read_rgb().tobytes())
            self.k += 1
        if self.k >= self.n_frames:
            if not self._closing:
                self._frames.put(None)
                self._closing = True
            if self._writer.is_alive() or self.proc.poll() is None:
                return True
            self._finish()
            return False
        return True

    def run_blocking(self, progress=None):
        self.start()
        while self.step(0.5):
            if self._frames.full() or self._closing:
                time.sleep(0.002)
            if progress:
                progress(self.k / max(self.n_frames, 1))
        if self.error:
            raise RuntimeError(self.error)
        return self.path

    def _finish(self):
        if not self._closing:
            self._frames.put(None)
            self._closing = True
        self._writer.join()
        rc = self.proc.wait()
        if rc != 0 or self._write_err is not None:
            self.error = f'ffmpeg exited with {rc}: {self._ffmpeg_err() or self._write_err}'
        self.done = True
        self._cleanup()

    def cancel(self):
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        if self.proc:
            while True:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    break
            self._frames.put(None)
        self._cleanup()
        try:
            os.remove(self.path)
        except OSError:
            pass

    def _cleanup(self):
        try:
            self.renderer_release()
        except Exception:
            pass
        try:
            os.remove(self.wav)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    def renderer_release(self):
        r = self.renderer
        for name in ('accum', 'b1a', 'b1b', 'b2a', 'b2b', 'out'):
            getattr(r, name + '_fbo').release()
            getattr(r, name + '_tex').release()

    @property
    def progress(self):
        return self.k / max(self.n_frames, 1)

    def eta(self):
        el = time.perf_counter() - self.started
        if self.k == 0:
            return None
        return el / self.k * (self.n_frames - self.k)


def probe(path):
    """(video frame count, video duration, audio duration) via ffprobe."""
    fp = ffprobe_path()
    if not fp:
        return None
    def q(args):
        return subprocess.run([fp, '-v', 'error', *args, path], capture_output=True, text=True,
                              creationflags=_NOWIN).stdout.strip()
    frames = q(['-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0'])
    vdur = q(['-select_streams', 'v:0', '-show_entries', 'stream=duration', '-of', 'csv=p=0'])
    adur = q(['-select_streams', 'a:0', '-show_entries', 'stream=duration', '-of', 'csv=p=0'])
    acodec = q(['-select_streams', 'a:0', '-show_entries', 'stream=codec_name', '-of', 'csv=p=0'])
    astart = q(['-select_streams', 'a:0', '-show_entries', 'stream=start_time', '-of', 'csv=p=0'])
    vstart = q(['-select_streams', 'v:0', '-show_entries', 'stream=start_time', '-of', 'csv=p=0'])

    def f(v):
        try:
            return float(v.split(',')[0])
        except (ValueError, IndexError):
            return None

    def tagdur(stream):
        v = q(['-select_streams', stream, '-show_entries', 'stream_tags=DURATION', '-of', 'csv=p=0'])
        try:
            h, m_, sec = v.split(':')
            return int(h) * 3600 + int(m_) * 60 + float(sec)
        except ValueError:
            return None
    if f(vdur) is None:
        vdur = str(tagdur('v:0'))
    if f(adur) is None:
        adur = str(tagdur('a:0'))
    return {'frames': int(f(frames) or 0), 'vdur': f(vdur), 'adur': f(adur), 'acodec': acodec,
            'astart': f(astart), 'vstart': f(vstart)}
