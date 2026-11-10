"""Audio decoding: libsndfile (WAV/FLAC/OGG/MP3/AIFF, any bit depth/rate) first, ffmpeg fallback.

Everything is converted to float32 stereo at the project rate (44.1 kHz)."""
import os
import shutil
import subprocess
from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly

PROJECT_RATE = 44100
AUDIO_EXTENSIONS = ['.flac', '.wav', '.mp3', '.ogg', '.oga', '.opus', '.m4a', '.aac', '.aif', '.aiff',
                    '.wma', '.alac', '.wv', '.ape', '.mka', '.webm', '.mp4']
DIALOG_FILTER = ['Audio files', ' '.join('*' + e for e in AUDIO_EXTENSIONS),
                 'FLAC', '*.flac', 'WAV', '*.wav', 'MP3', '*.mp3', 'All files', '*']


@dataclass
class AudioInfo:
    path: str
    source_rate: int
    source_channels: int
    subtype: str
    decoder: str
    duration: float


_BUNDLED_FFMPEG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'tools', 'ffmpeg', 'bin', 'ffmpeg.exe')


def find_ffmpeg():
    # Bundled 8.1 first: its NVENC works with driver 581.x, the system 9.x needs driver 610+.
    exe = os.environ.get('FDS_FFMPEG')
    if exe and os.path.exists(exe):
        return exe
    if os.path.exists(_BUNDLED_FFMPEG):
        return _BUNDLED_FFMPEG
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    base = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'WinGet', 'Packages')
    if os.path.isdir(base):
        for root, _dirs, files in os.walk(base):
            if 'ffmpeg.exe' in files:
                return os.path.join(root, 'ffmpeg.exe')
    return None


_C = 0.70710678


def downmix_to_stereo(x):
    """x (n, ch) float32 -> (n, 2). Uses ITU-style coefficients for common layouts (WAV/FLAC order)."""
    ch = x.shape[1]
    if ch == 1:
        return np.repeat(x, 2, axis=1)
    if ch == 2:
        return x
    if ch == 3:      # L R C
        L = x[:, 0] + _C * x[:, 2]
        R = x[:, 1] + _C * x[:, 2]
        norm = 1 + _C
    elif ch == 4:    # L R Ls Rs
        L = x[:, 0] + _C * x[:, 2]
        R = x[:, 1] + _C * x[:, 3]
        norm = 1 + _C
    elif ch == 5:    # L R C Ls Rs
        L = x[:, 0] + _C * x[:, 2] + _C * x[:, 3]
        R = x[:, 1] + _C * x[:, 2] + _C * x[:, 4]
        norm = 1 + 2 * _C
    elif ch == 6:    # L R C LFE Ls Rs
        L = x[:, 0] + _C * x[:, 2] + _C * x[:, 4]
        R = x[:, 1] + _C * x[:, 2] + _C * x[:, 5]
        norm = 1 + 2 * _C
    elif ch == 8:    # L R C LFE Lb Rb Ls Rs
        L = x[:, 0] + _C * x[:, 2] + _C * x[:, 4] + _C * x[:, 6]
        R = x[:, 1] + _C * x[:, 2] + _C * x[:, 5] + _C * x[:, 7]
        norm = 1 + 3 * _C
    else:
        L = x[:, 0::2].mean(axis=1)
        R = x[:, 1::2].mean(axis=1)
        norm = 1.0
    return (np.stack([L, R], axis=1) / norm).astype(np.float32)


def resample(x, src, dst):
    if src == dst:
        return x
    g = gcd(int(src), int(dst))
    return resample_poly(x, dst // g, src // g, axis=0).astype(np.float32)


def _decode_soundfile(path):
    import soundfile as sf
    info = sf.info(path)
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    return data, int(sr), info.channels, info.subtype


def _decode_ffmpeg(path, rate):
    exe = find_ffmpeg()
    if not exe:
        raise RuntimeError('ffmpeg not found')
    probe = subprocess.run([exe, '-hide_banner', '-i', path], capture_output=True, text=True,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    src_rate, src_ch = rate, 2
    for line in probe.stderr.splitlines():
        if 'Audio:' in line:
            parts = [p.strip() for p in line.split(',')]
            for p in parts:
                if p.endswith(' Hz'):
                    try:
                        src_rate = int(p.split()[0])
                    except ValueError:
                        pass
                if p.startswith(('mono',)):
                    src_ch = 1
                elif p.startswith(('5.1', '6 channels')):
                    src_ch = 6
                elif p.startswith(('7.1', '8 channels')):
                    src_ch = 8
            break
    out = subprocess.run([exe, '-v', 'error', '-i', path, '-vn', '-f', 'f32le', '-acodec', 'pcm_f32le',
                          '-ac', '2', '-ar', str(rate), '-'], capture_output=True,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError(out.stderr.decode(errors='replace')[-400:] or 'ffmpeg decode failed')
    data = np.frombuffer(out.stdout, dtype=np.float32).reshape(-1, 2).copy()
    return data, src_rate, src_ch


def load_audio(path, rate=PROJECT_RATE):
    """Returns (samples float32 (n,2) at `rate`, AudioInfo)."""
    errors = []
    try:
        data, sr, ch, subtype = _decode_soundfile(path)
        stereo = resample(downmix_to_stereo(data), sr, rate)
        info = AudioInfo(path, sr, ch, subtype, 'libsndfile', len(stereo) / rate)
        return np.ascontiguousarray(stereo, dtype=np.float32), info
    except Exception as e:  # unsupported container/codec -> ffmpeg
        errors.append(f'libsndfile: {e}')
    try:
        data, sr, ch = _decode_ffmpeg(path, rate)
        return data, AudioInfo(path, sr, ch, 'ffmpeg', 'ffmpeg', len(data) / rate)
    except Exception as e:
        errors.append(f'ffmpeg: {e}')
    raise RuntimeError(f'Could not decode {os.path.basename(path)}: ' + ' | '.join(errors))


def envelope(x, rate, per_second=200):
    """Peak envelope (frames, 2) of abs mono, used for the timeline waveform."""
    hop = max(1, rate // per_second)
    mono = np.abs(x).max(axis=1)
    n = len(mono) // hop
    if n == 0:
        return np.zeros((1,), np.float32)
    return mono[:n * hop].reshape(n, hop).max(axis=1).astype(np.float32)
