"""Timeline audio playback with a DAC-accurate clock (visuals follow what is actually heard)."""
import threading
import time

import numpy as np

SR = 44100


class Player:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()
        self.playing = False
        self.pos = 0                 # next sample index to be sent to the device
        self._anchor = None          # (dac_time, sample_pos) of the most recent callback
        self._paused_t = 0.0
        self.stream = None
        self.volume = 1.0
        self.muted = False
        self.loop = False
        self.ended = False
        self.device_ok = True
        self._wall = None
        self._segments = []
        self.alt_segments = None     # dots -> sound buffers [(start, int16 (n, 2))]
        self.use_alt = False

    def _build_segments(self):
        offs, total = self.engine.offsets()
        segs = []
        for o, ta in zip(offs, self.engine.tracks()):
            if ta.audio is not None:
                segs.append((int(round(o * SR)), ta.audio))
        self._segments = self.alt_segments if (self.use_alt and self.alt_segments is not None) else segs
        self._total = int(round(total * SR))

    def _callback(self, outdata, frames, time_info, status):
        with self.lock:
            pos = self.pos
            self.pos += frames
            self._anchor = (time_info.outputBufferDacTime, pos)
        out = np.zeros((frames, 2), np.float32)
        for start, arr in self._segments:
            a = max(pos, start)
            b = min(pos + frames, start + len(arr))
            if b > a:
                out[a - pos:b - pos] = arr[a - start:b - start].astype(np.float32) * (1.0 / 32767.0)
        g = 0.0 if self.muted else self.volume
        outdata[:] = out * g
        if pos >= self._total:
            self.ended = True

    def play(self, t):
        self.stop()
        self._build_segments()
        self.pos = max(0, int(round(t * SR)))
        self.ended = False
        self._paused_t = t
        try:
            import sounddevice as sd
            self.stream = sd.OutputStream(samplerate=SR, channels=2, dtype='float32', callback=self._callback,
                                          latency='low')
            self.stream.start()
            self.device_ok = True
        except Exception:
            self.stream = None
            self.device_ok = False
            self._wall = (time.perf_counter(), t)
        self.playing = True

    def stop(self):
        t = self.time()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.playing = False
        self._anchor = None
        self._paused_t = t
        return t

    def seek(self, t):
        if self.playing:
            self.play(t)
        else:
            self._paused_t = max(0.0, t)

    def latency(self):
        """Output latency reported by the audio device (s)."""
        try:
            return float(self.stream.latency) if self.stream is not None else 0.0
        except Exception:
            return 0.0

    def submitted_time(self):
        """Position of the audio most recently handed to the device (ignores output latency)."""
        return self.time() + self.latency() if self.playing else self.time()

    def time(self):
        """Position (s) of the audio currently leaving the speakers."""
        if not self.playing:
            return self._paused_t
        if self.stream is None:
            t0, start = self._wall
            return start + (time.perf_counter() - t0)
        with self.lock:
            anchor = self._anchor
        if anchor is None:
            return self._paused_t
        dac, p = anchor
        now = self.stream.time
        return max(0.0, (p + (now - dac) * SR) / SR)
