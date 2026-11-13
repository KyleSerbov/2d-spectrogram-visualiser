"""Dots -> sound: rebuild audio from the dots the visualiser draws.

Two decoders:
* exact  - every dot keeps its draw order, i.e. which FFT bin of which band it came from, so its exact
           frequency is known. The drawn dots' L/R levels are put back into those bins and resynthesised
           with one inverse FFT per analysis frame, overlap-added.
* screen - only what a viewer could read off the picture: y -> pitch (quantised to pixel rows),
           x -> pan, pixel brightness (dot intensity, optionally times colour luminance) -> loudness in dB.

Dots carry no phase, so it is invented:
* locked - every spectral peak's true frequency is estimated (quadratic interpolation), its phase is
           carried from frame to frame at that frequency, and every bin around it takes the peak's phase,
           so each peak comes back as one clean sinusoid (closest to the original);
* free   - every bin runs at its own centre frequency (classic phase-vocoder phasiness, chorus-like);
* noise  - random phase every frame (whispery).

Only dots that are actually drawn make sound: threshold, peak picking, max dots and colour modes that
hide or dim dots all apply. Frame f sounds at the time the viewer shows it (track time f * hop).

Dots inserted between dots (Render > Insert between dots) have no bin of their own: their screen height
gives a frequency, i.e. a fractional draw order. The screen decoder reads them like any other dot; the
exact decoder plays them on a fine 2.7 Hz grid with levels blended from their two parent dots
(scaled by 1/sqrt(points per gap) so the added energy stays comparable to the original dots)."""
import hashlib
import json
import threading
import time
import traceback

import numpy as np

from . import analysis as A

SR = A.SR
SONIFY_DEFAULTS = {'source': 'exact', 'phase': 'locked', 'stereo': 'original', 'colour_weight': 0.0,
                   'range_db': 48.0, 'rows': 1080, 'gain_db': 0.0, 'insert_db': 0.0}
SOURCES = [('exact', 'Exact (dot draw order -> frequency bin)'), ('screen', 'Screen (decode the picture)')]
PHASES = [('locked', 'Clean (peak-locked phase)'), ('free', 'Chorus (free-running phase)'),
          ('noise', 'Whisper (random phase)')]
STEREO = [('original', "Each dot's original L/R level"), ('dot', "Dot's x position")]
SCREEN_NFFT = 16384
SCREEN_LEVEL = 0.02
TWO_PI = 2.0 * np.pi
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
CHUNK = 64


def sanitize(s):
    out = dict(SONIFY_DEFAULTS)
    out.update({k: v for k, v in (s or {}).items() if k in SONIFY_DEFAULTS})
    for key, choices in (('source', SOURCES), ('phase', PHASES), ('stereo', STEREO)):
        if out[key] not in [c[0] for c in choices]:
            out[key] = SONIFY_DEFAULTS[key]
    out['colour_weight'] = float(np.clip(out['colour_weight'], 0.0, 1.0))
    out['range_db'] = float(np.clip(out['range_db'], 6.0, 120.0))
    out['rows'] = int(np.clip(out['rows'], 16, 8640))
    out['gain_db'] = float(np.clip(out['gain_db'], -40.0, 24.0))
    out['insert_db'] = float(np.clip(out['insert_db'], -60.0, 12.0))
    return out


def state_key(project, engine):
    """Hash of everything the dots audio depends on (compare with Sonifier.key to detect stale audio)."""
    d = project.to_dict()
    keep = {k: d[k] for k in ('tracks', 'keyframes', 'render', 'analysis', 'align', 'gap_s')}
    keep['sonify'] = sanitize(getattr(project, 'sonify', None))
    need = project.required_analyses()
    keep['spec'] = [(getattr(ta, 'spectrum_key', None), ta.status if need else '') for ta in engine.tracks()]
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


class Grid:
    """Bins [k_lo, k_hi) of an nfft-point FFT, resynthesised as L-sample grains every hop samples."""

    def __init__(self, nfft, k_lo, k_hi, L, hop, phase, seed):
        self.nfft, self.k_lo, self.k_hi, self.L, self.hop, self.phase = nfft, k_lo, k_hi, int(L), hop, phase
        self.K = k_hi - k_lo
        self.seed = seed
        self.phi0 = np.random.default_rng(seed).uniform(0, TWO_PI, self.K)
        self.kk = np.arange(k_lo, k_hi, dtype=np.float64)
        self.win = 0.5 - 0.5 * np.cos(TWO_PI * np.arange(self.L) / self.L)
        self._twin = None
        self.gain = 1.0
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed + 1)
        self.pk = np.zeros(0, np.int64)
        self.pk_f = np.zeros(0)
        self.pk_ph = np.zeros(0)

    def phases(self, M, t):
        """Phase of every bin for the grain centred at sample t. M: (K,) magnitudes (L+R)."""
        K = self.K
        if self.phase == 'noise':
            return self.rng.uniform(0, TWO_PI, K)
        if self.phase == 'free':
            return (self.kk * (TWO_PI * (t % self.nfft) / self.nfft) + self.phi0) % TWO_PI
        left = np.r_[-1.0, M[:-1]]
        right = np.r_[M[1:], -1.0]
        p = np.flatnonzero((M > 0) & (M >= left) & (M > right))
        if len(p) == 0:
            self.pk, self.pk_f, self.pk_ph = p, np.zeros(0), np.zeros(0)
            return np.zeros(K)
        # true peak frequency in bins: quadratic interpolation of log magnitude (needs both neighbours drawn)
        pl, pr = np.maximum(p - 1, 0), np.minimum(p + 1, K - 1)
        ok = (p > 0) & (p < K - 1) & (M[pl] > 0) & (M[pr] > 0)
        lm = np.log(np.maximum(M, 1e-30))
        a, b, c = lm[pl], lm[p], lm[pr]
        den = a - 2 * b + c
        d = np.where(ok & (den < -1e-12), 0.5 * (a - c) / np.where(den < -1e-12, den, -1.0), 0.0)
        fbin = self.k_lo + p + np.clip(d, -0.5, 0.5)
        # new peaks start at an absolute-time phase; continuing peaks advance at their mean frequency
        ph = (((fbin * t) / self.nfft) % 1.0) * TWO_PI + self.phi0[p]
        if len(self.pk):
            j = np.searchsorted(self.pk, p)
            jl, jr = np.clip(j - 1, 0, len(self.pk) - 1), np.clip(j, 0, len(self.pk) - 1)
            pick = np.where(np.abs(self.pk[jl] - p) <= np.abs(self.pk[jr] - p), jl, jr)
            near = np.abs(self.pk[pick] - p) <= max(3, self.nfft // 4096)
            adv = self.pk_ph[pick] + TWO_PI * 0.5 * (self.pk_f[pick] + fbin) * self.hop / self.nfft
            ph = np.where(near, adv, ph)
        ph %= TWO_PI
        self.pk, self.pk_f, self.pk_ph = p, fbin, ph
        # every bin takes the phase of its nearest peak
        idx = np.arange(K)
        j = np.searchsorted(p, idx)
        jl, jr = np.clip(j - 1, 0, len(p) - 1), np.clip(j, 0, len(p) - 1)
        own = np.where(np.abs(p[jl] - idx) <= np.abs(p[jr] - idx), jl, jr)
        return ph[own]

    def grains(self, AL, AR, PH):
        """AL, AR, PH: (C, K). Returns torch (2, C, L) windowed grains; grain sample L/2 = frame centre."""
        import torch
        dev = A.DEVICE
        if self._twin is None:
            self._twin = torch.from_numpy(self.win.astype(np.float32)).to(dev)
        C = AL.shape[0]
        amp = torch.from_numpy(np.stack([AL, AR]).astype(np.float32)).to(dev) * float(self.nfft / 2 * self.gain)
        ph = torch.from_numpy(PH.astype(np.float32)).to(dev)
        X = torch.zeros((2, C, self.nfft // 2 + 1), dtype=torch.complex64, device=dev)
        X[:, :, self.k_lo:self.k_hi] = torch.complex(amp * torch.cos(ph), amp * torch.sin(ph))
        x = torch.fft.irfft(X, n=self.nfft)
        h = self.L // 2
        return torch.cat([x[..., self.nfft - h:], x[..., :self.L - h]], dim=-1) * self._twin

    def calibrate(self, pattern):
        """Scale so a steady sine of amplitude 1 (whose dots look like `pattern`) comes back at amplitude 1."""
        self.gain = 1.0
        self.reset()
        row = np.zeros(self.K)
        c, half = self.K // 2, len(pattern) // 2
        for i, v in enumerate(pattern):
            k = c - half + i
            if 0 <= k < self.K:
                row[k] = v
        n = int(min(3 * self.L / self.hop + 8, 600))
        PH = np.stack([self.phases(2 * row, f * self.hop) for f in range(n)])
        g = self.grains(np.tile(row, (n, 1)), np.zeros((n, self.K)), PH)[0].cpu().numpy()
        buf = np.zeros((n - 1) * self.hop + self.L)
        for f in range(n):
            buf[f * self.hop:f * self.hop + self.L] += g[f]
        mid = buf[self.L:(n - 1) * self.hop]
        rms = float(np.sqrt(np.mean(mid ** 2))) if len(mid) else 0.0
        self.gain = (1 / np.sqrt(2)) / max(rms, 1e-12)
        self.reset()


def _limit(y):
    """Transparent below 0.9, smoothly saturating to 1.0 above."""
    a = np.abs(y)
    over = a > 0.9
    if over.any():
        y = y.copy()
        y[over] = np.sign(y[over]) * (0.9 + 0.1 * np.tanh((a[over] - 0.9) / 0.1))
    return y


class Sonifier:
    """Background render of the whole timeline's dots into per-track int16 stereo buffers.

    `segments` [(start_sample, int16 (n, 2))] can be played while rendering (unrendered parts are silent)."""

    def __init__(self, engine, settings, key=None, aspect=None):
        self.engine = engine
        self.s = sanitize(settings)
        self.key = key
        self.aspect = aspect
        self.paths = list(engine.project.tracks)
        self.tracks = engine.tracks()
        offs, total = engine.offsets()
        self.total_samples = int(round(total * SR))
        self.out = [np.zeros((len(ta.audio) if ta.audio is not None else 0, 2), np.int16) for ta in self.tracks]
        self.segments = [(int(round(o * SR)), arr) for o, arr in zip(offs, self.out)]
        self.progress = 0.0
        self.done = False
        self.error = None
        self.cancelled = False
        self.limited = 0
        self.started = time.perf_counter()
        self.elapsed = 0.0
        self._grids = {}
        self.uses_insert = engine.project.uses_insert()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def cancel(self):
        if not self.done:
            self.cancelled = True

    def wait(self):
        self._thread.join()
        return self

    @property
    def running(self):
        return self._thread.is_alive()

    def eta(self):
        el = time.perf_counter() - self.started
        return el / self.progress * (1 - self.progress) if 0.002 < self.progress < 1 else None

    # ------------------------------------------------------------------ render
    def _run(self):
        try:
            total = sum(ta.spectrum.n_frames for ta in self.tracks if ta.spectrum is not None) or 1
            done = 0
            for ti, ta in enumerate(self.tracks):
                if ta.spectrum is None or ta.audio is None:
                    continue
                done = self._track(ti, ta, done, total)
                if self.cancelled:
                    return
            self.progress = 1.0
            self.done = True
        except Exception as e:
            traceback.print_exc()
            self.error = f'{type(e).__name__}: {e}'
        finally:
            self.elapsed = time.perf_counter() - self.started

    def _make_grids(self, ta, render):
        sp = ta.spectrum
        key = (json.dumps(sp.settings, sort_keys=True), render['freq_min'], render['freq_max'])
        if key in self._grids:
            return self._grids[key]
        hop = sp.hop
        grids = []
        fine_L = min(SCREEN_NFFT, max(4 * hop, 2048))
        if self.s['source'] == 'screen':
            g = Grid(SCREEN_NFFT, 1, SCREEN_NFFT // 2, fine_L, hop, self.s['phase'], 7)
            g.calibrate(np.ones(1))
            grids.append((g, 'screen'))
        else:
            if self.uses_insert:
                # inserted dots sit at fractional draw-order positions: a fine grid (2.7 Hz bins) carries them
                g = Grid(SCREEN_NFFT, 1, SCREEN_NFFT // 2, fine_L, hop, self.s['phase'], 5)
                g.calibrate(np.ones(1))
                grids.append((g, 'insert'))
            start = 0
            for bi, b in enumerate(sp.bands):
                m = (b.freqs >= render['freq_min']) & (b.freqs <= render['freq_max'])
                cnt = int(m.sum())
                if cnt:
                    k = np.round(b.freqs[m] * b.nfft / SR).astype(np.int64)
                    g = Grid(b.nfft, int(k[0]), int(k[-1]) + 1, min(b.N, b.nfft), hop, self.s['phase'], 11 + bi)
                    W = np.abs(np.fft.rfft(A.get_window(sp.settings['window_fn'], b.N), b.nfft))
                    W /= W[0]
                    D = 4 * max(1, b.nfft // b.N)
                    g.calibrate(np.r_[W[D:0:-1], W[:D + 1]])
                    grids.append((g, (start, start + cnt)))
                start += cnt
        self._grids[key] = grids
        return grids

    @staticmethod
    def _bin_of_y(y, render, grid):
        """Screen height -> frequency -> (fractional draw order rounded to) a bin of the fine grid."""
        from .project import y_to_freq
        fr = y_to_freq(y, render['freq_min'], render['freq_max'], render['freq_scale'])
        return np.clip(np.round(fr * grid.nfft / SR).astype(np.int64), grid.k_lo, grid.k_hi - 1) - grid.k_lo

    def _screen_amps(self, x, y, inten, rgb, render, grid):
        s = self.s
        K = grid.K
        aL, aR = np.zeros(K), np.zeros(K)
        vis = inten > 0
        if not vis.any():
            return aL, aR
        v = inten[vis]
        cw = s['colour_weight']
        if cw > 0:
            v = v * ((1 - cw) + cw * (np.clip(rgb[vis], 0, 1) @ LUMA))
        v = np.clip(v, 0, 1)
        amp = np.where(v > 0, 10 ** ((v - 1) * s['range_db'] / 20), 0.0) * SCREEN_LEVEL
        rows = s['rows']
        k = self._bin_of_y((np.floor(y[vis] * rows) + 0.5) / rows, render, grid)
        xv = -x[vis] if render['flip_x'] else x[vis]
        th = (np.clip(xv, -1, 1) + 1) * (np.pi / 4)
        np.add.at(aL, k, amp * np.cos(th) * np.sqrt(2))
        np.add.at(aR, k, amp * np.sin(th) * np.sqrt(2))
        return aL, aR

    def _insert_amps(self, ins, full, render, grid):
        """Exact decoder: an inserted dot's level is blended from its two parent dots' levels; its frequency
        comes from where it sits on screen."""
        K = grid.K
        aL, aR = np.zeros(K), np.zeros(K)
        if ins is None:
            return aL, aR
        fl, fr = full
        a, b, t = ins['a'], ins['b'], ins['t']
        n = max(1, int(round(render.get('insert_count', 1))))
        g = float(render.get('insert_brightness', 1.0)) / np.sqrt(n) * 10 ** (self.s['insert_db'] / 20)
        k = self._bin_of_y(ins['y'], render, grid)
        np.add.at(aL, k, (fl[a] + (fl[b] - fl[a]) * t) * g)
        np.add.at(aR, k, (fr[a] + (fr[b] - fr[a]) * t) * g)
        return aL, aR

    def _amps(self, ctx, render, inten, gain, rgb):
        """Exact decoder: L/R level of every dot in draw order (0 where not drawn)."""
        s = self.s
        vis = inten > 0
        cw = s['colour_weight']
        w = np.clip(gain, 0, 4) * vis
        if cw > 0:
            w = w * ((1 - cw) + cw * (np.clip(rgb, 0, 1) @ LUMA))
        if s['stereo'] == 'dot':
            m = ctx.amp * w
            x = -ctx.x if render['flip_x'] else ctx.x
            th = (np.clip(x, -1, 1) + 1) * (np.pi / 4)
            return m * np.cos(th) * np.sqrt(2), m * np.sin(th) * np.sqrt(2)
        return ctx.L * w, ctx.R * w

    def _write(self, out, block, lo, gain):
        n = len(out)
        a, b = max(lo, 0), min(lo + block.shape[1], n)
        if b <= a:
            return
        y = block[:, a - lo:b - lo].T * gain
        self.limited += int((np.abs(y) > 0.9).sum())
        out[a:b] = np.round(_limit(y) * 32767).astype(np.int16)

    def _track(self, ti, ta, done, total):
        import torch
        eng = self.engine
        sp = ta.spectrum
        hop, nF = sp.hop, sp.n_frames
        out = self.out[ti]
        render0 = eng.project.render_at(eng.track_offset(ti))
        grids = self._make_grids(ta, render0)
        for g, _ in grids:
            g.reset()
        h = max(g.L for g, _ in grids) // 2
        gain = 10 ** (self.s['gain_db'] / 20)
        pend, pend_lo = np.zeros((2, 0), np.float32), -h
        for f0 in range(0, nF, CHUNK):
            if self.cancelled:
                return done
            if list(eng.project.tracks) != self.paths:
                self.cancelled = True
                self.error = 'the track list changed while rendering; render again'
                return done
            f1 = min(nF, f0 + CHUNK)
            C = f1 - f0
            rows = [(np.zeros((C, g.K)), np.zeros((C, g.K)), np.zeros((C, g.K))) for g, _ in grids]
            for f in range(f0, f1):
                with eng.lock:
                    ctx, render, inten, gn, rgb, ins = eng.frame_grid(ti, f, self.aspect)
                t = f * hop
                full = self._amps(ctx, render, inten, gn, rgb) if (ctx.n and self.s['source'] == 'exact') else None
                for (g, kind), (AL, AR, PH) in zip(grids, rows):
                    if ctx.n == 0:
                        PH[f - f0] = g.phases(np.zeros(g.K), t)
                        continue
                    if kind == 'screen':
                        x, y, iv, cl = ctx.x, ctx.y, inten, rgb
                        if ins is not None:
                            x, y = np.concatenate([x, ins['x']]), np.concatenate([y, ins['y']])
                            iv, cl = np.concatenate([iv, ins['inten']]), np.concatenate([cl, ins['rgb']])
                        l, r = self._screen_amps(x, y, iv, cl, render, g)
                    elif kind == 'insert':
                        l, r = self._insert_amps(ins, full, render, g)
                    else:
                        l, r = full[0][kind[0]:kind[1]], full[1][kind[0]:kind[1]]
                    AL[f - f0], AR[f - f0] = l, r
                    PH[f - f0] = g.phases(l + r, t)
            lo, hi = f0 * hop - h, (f1 - 1) * hop + h
            with torch.no_grad():
                buf = torch.zeros((2, hi - lo), dtype=torch.float32, device=A.DEVICE)
                for (g, _), (AL, AR, PH) in zip(grids, rows):
                    seg = g.grains(AL, AR, PH)
                    starts = torch.arange(f0, f1, device=A.DEVICE) * hop - g.L // 2 - lo
                    idx = (starts[:, None] + torch.arange(g.L, device=A.DEVICE)[None, :]).reshape(-1)
                    buf.index_add_(1, idx, seg.reshape(2, -1))
                chunk = buf.cpu().numpy()
            need = hi - pend_lo
            if pend.shape[1] < need:
                pend = np.pad(pend, ((0, 0), (0, need - pend.shape[1])))
            pend[:, lo - pend_lo:hi - pend_lo] += chunk
            fin = f1 * hop - h
            self._write(out, pend[:, :fin - pend_lo], pend_lo, gain)
            pend, pend_lo = pend[:, fin - pend_lo:], fin
            done += C
            self.progress = min(done / total, 0.999)
        self._write(out, pend, pend_lo, gain)
        return done


def mix_segments(segments, a, b):
    """Float32 (b - a, 2) mix of int16 segments over samples [a, b)."""
    out = np.zeros((b - a, 2), np.float32)
    for s0, arr in segments:
        x0, x1 = max(a, s0), min(b, s0 + len(arr))
        if x1 > x0:
            out[x0 - a:x1 - a] += arr[x0 - s0:x1 - s0].astype(np.float32) / 32767.0
    return out


def write_audio(path, segments, total_samples):
    """Write the segments as a 16-bit WAV (or FLAC by extension) of exactly total_samples."""
    import soundfile as sf
    fmt = 'FLAC' if path.lower().endswith('.flac') else 'WAV'
    with sf.SoundFile(path, 'w', SR, 2, subtype='PCM_16', format=fmt) as f:
        for a in range(0, total_samples, SR * 10):
            f.write(mix_segments(segments, a, min(total_samples, a + SR * 10)))
    return path
