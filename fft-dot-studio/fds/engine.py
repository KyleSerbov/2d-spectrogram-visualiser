"""Frame contexts, colour evaluation with keyframe tweening, and deterministic trail gathering.

Given a timeline time t, the visible image is a pure function of (project, analysis, t): the trail is
rebuilt from the last N analysis frames with exponential weights, never from a feedback buffer, so
scrubbing and offline export are exact."""
import threading
from collections import OrderedDict

import numpy as np

from . import analysis as A
from .colorspace import mix_oklab
from .modes import MODES, ModeOut, Pal, greedy_peaks, interpolate_params
from .palettes import mix_luts
from .project import align_shifts, freq_to_y

CTX_KEYS = ('freq_min', 'freq_max', 'freq_scale', 'amp_scale', 'db_floor', 'normalise', 'running_release_s',
            'threshold', 'peak_pick', 'peak_min_dist', 'smooth_attack_ms', 'smooth_release_ms', 'pan_mode',
            'pan_spread', 'edge_margin')


class LRU(OrderedDict):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap

    def get_(self, k):
        v = self.get(k)
        if v is not None:
            self.move_to_end(k)
        return v

    def put(self, k, v):
        self[k] = v
        self.move_to_end(k)
        while len(self) > self.cap:
            self.popitem(last=False)


class FrameCtx:
    """Everything a colour mode may look at for one analysis frame of one track."""

    def __init__(self, engine, ti, f, ta, render, aspect):
        self.engine, self.ti, self.f, self.ta = engine, ti, f, ta
        sp = ta.spectrum
        self.hop_s = sp.hop / A.SR
        self.local_t = f * self.hop_s
        self.t = engine.track_offset(ti) + self.local_t
        self.aspect = aspect
        self.render = render
        L, R, ipd = engine.frame_lr(ti, ta, f, render)
        freq = sp.freqs
        fmin, fmax = render['freq_min'], render['freq_max']
        sel = (freq >= fmin) & (freq <= fmax)
        # contiguous per-band slices after selection
        self.band_slices = []
        start = 0
        for b in sp.bands:
            m = (b.freqs >= fmin) & (b.freqs <= fmax)
            cnt = int(m.sum())
            if cnt:
                self.band_slices.append((start, start + cnt))
            start += cnt
        self.freq = freq[sel]
        L, R, ipd = L[sel], R[sel], ipd[sel]
        self.L, self.R = L, R
        self.n = len(self.freq)
        self.amp = 0.5 * (L + R)
        fm = float(np.maximum(L, R).max()) if self.n else 0.0
        self.frame_max = fm + 1e-12
        if render['normalise'] == 'running':
            fs = int(round(f + engine.shifts(ta)[0][0]))
            ref = engine.running_ref(ti, ta, render['running_release_s'])[min(max(fs, 0), ta.spectrum.n_frames - 1)]
        elif render['normalise'] == 'global':
            ref = ta.spectrum.ref_global
        else:
            ref = self.frame_max
        self.ref = max(float(ref), 1e-9)
        self.amp_rel = self.amp / self.frame_max
        rel = self.amp / self.ref
        if render['amp_scale'] == 'db':
            floor = render['db_floor']
            self.level = np.clip((20 * np.log10(np.maximum(rel, 1e-12)) - floor) / -floor, 0, 1).astype(np.float32)
        else:
            self.level = rel.astype(np.float32)
        self.threshold = render['threshold']
        self.visible = rel > render['threshold']
        if render['pan_mode'] == 'phase':
            self.pan = (ipd / np.pi).astype(np.float32)
        else:
            self.pan = ((R - L) / (R + L + 1e-9)).astype(np.float32)
        m = render['edge_margin']
        self.x = np.clip(self.pan * render['pan_spread'], -1 + m, 1 - m).astype(np.float32)
        self.y = np.clip(freq_to_y(self.freq, fmin, fmax, render['freq_scale']), m / 2, 1 - m / 2).astype(np.float32)
        if render['peak_pick'] and self.n:
            from .modes import band_peaks
            pk = band_peaks(self, np.where(self.visible, self.amp, 0), None, None)
            keep = greedy_peaks(self.x, self.y, self.amp, np.isin(np.arange(self.n), pk), render['peak_min_dist'])
            vis = np.zeros(self.n, bool)
            vis[keep] = True
            self.visible &= vis
        self.feat = ta.feats
        self.fi = ta.feat_index(self.local_t + engine.shifts(ta)[1])

    # ---- history
    def prev(self, k):
        if k == 0:
            return self
        if self.f - k < 0:
            return None
        return self.engine.ctx(self.ti, self.f - k, self.render, self.aspect)

    # ---- features
    def feat_val(self, name, smooth=0):
        arr = self.feat[name]
        a = max(0, self.fi - int(smooth))
        return float(np.mean(arr[a:self.fi + 1]))

    def feat_window(self, name, secs):
        arr = self.feat[name]
        a = max(0, self.fi - int(secs * A.FEAT_FPS))
        return arr[a:self.fi + 1].mean(axis=0)

    def feat_cumsum(self, name):
        cs = self.engine.cumsum(self.ti, self.ta, name)
        return float(cs[self.fi])

    def onset_decay(self, decay_s, bands=False):
        arr = self.feat['onset_bands'] if bands else self.feat['onset']
        J = int(np.ceil(4 * decay_s * A.FEAT_FPS)) + 1
        a = max(0, self.fi - J + 1)
        seg = arr[a:self.fi + 1]
        ages = (self.fi - np.arange(a, self.fi + 1)) / A.FEAT_FPS
        w = np.exp(-ages / max(decay_s, 1e-3))
        if bands:
            return (seg * w[:, None]).max(axis=0)
        return float((seg * w).max())

    def beat_pos(self):
        if self.feat is None:
            return self.t * 2.0
        beats = self.feat['beats']
        tempo = float(self.feat['tempo'][0]) or 120.0
        per = 60.0 / tempo
        t = self.local_t
        if len(beats) < 2:
            return t / per
        i = int(np.searchsorted(beats, t, side='right')) - 1
        if i < 0:
            return (t - beats[0]) / per
        if i >= len(beats) - 1:
            return (len(beats) - 1) + (t - beats[-1]) / per
        return i + (t - beats[i]) / max(beats[i + 1] - beats[i], 1e-3)

    def beat_flash(self, decay_beats=0.2):
        ph = self.beat_pos() % 1.0
        return float(np.exp(-ph / max(decay_beats, 1e-3)))

    def stem_ratios(self):
        st = self.ta.stems
        if st is None:
            return None
        row = np.asarray(st[min(self.fi, st.shape[0] - 1)], dtype=np.float32) / 255.0
        idx = np.clip(np.round(self.freq / (A.SR / A.FEAT_N)).astype(np.int64), 0, row.shape[0] - 1)
        return row[idx]

    def vowel_state(self, hold=True):
        v = self.ta.vowel
        if v is None:
            return None
        i = min(self.fi, len(v['voiced']) - 1)
        if hold:
            from .modes import _vowel_hold
            w = _vowel_hold(self.ta)[i]
        else:
            w = v['weights'][i]
        return np.asarray(w, np.float64), float(v['voiced'][i]), float(v['openness'][i])


class Layer:
    def __init__(self, mode, params, pal, use_palette, name):
        self.mode, self.params, self.pal, self.use_palette, self.name = mode, params, pal, use_palette, name


class Engine:
    def __init__(self, project, manager, aspect=16 / 9):
        self.project = project
        self.manager = manager
        self.aspect = aspect
        self._ctx = LRU(1500)
        self._dots = LRU(4000)
        self._smooth = LRU(3000)
        self._running = {}
        self._cumsum = {}
        self._lut_mix = LRU(128)
        self.lock = threading.RLock()

    # ---------------------------------------------------------------- timeline
    def tracks(self):
        return [self.manager.get(p) for p in self.project.tracks]

    def durations(self):
        return [ta.duration for ta in self.tracks()]

    def offsets(self):
        return self.project.track_offsets(self.durations())

    def export_blockers(self):
        """Analyses the timeline's colour modes still need, as 'track: stage' strings (empty = can export)."""
        need = self.project.required_analyses()
        out = []
        for ta in self.tracks():
            if ta.status == 'error':
                out.append(f'{ta.name}: analysis failed ({ta.error})')
            elif ta.audio is None or ta.spectrum is None or ta.status in ('queued', 'decoding', 'spectrum'):
                out.append(f'{ta.name}: spectrum')
            elif 'feats' in need and ta.feats is None:
                out.append(f'{ta.name}: beat/onset features')
            elif need & {'stems', 'vowel'} and ta.status != 'ready':
                out.append(f'{ta.name}: ' + ' + '.join(sorted(need & {'stems', 'vowel'})))
        return out

    def total_duration(self):
        return self.offsets()[1]

    def track_offset(self, ti):
        return self.offsets()[0][ti]

    def locate(self, t):
        offs, _total = self.offsets()
        tas = self.tracks()
        for i, (o, ta) in enumerate(zip(offs, tas)):
            if o <= t < o + ta.duration:
                return i, t - o
        return None, None

    # ---------------------------------------------------------------- analysis access
    def shifts(self, ta):
        sp = ta.spectrum
        return align_shifts(self.project.align, [b.N for b in sp.bands], sp.hop, A.SR, A.FEAT_N)

    def raw_frame(self, ta, f):
        return ta.spectrum.get_frame_shifted(f, self.shifts(ta)[0])

    def frame_lr(self, ti, ta, f, render):
        att, rel = render['smooth_attack_ms'], render['smooth_release_ms']
        if att <= 0 and rel <= 0:
            fr = self.raw_frame(ta, f)
            return fr['L'], fr['R'], fr['ipd']
        return self._smoothed(ti, ta, f, att, rel)

    def _smoothed(self, ti, ta, f, att, rel):
        sp = ta.spectrum
        sh = self.shifts(ta)[0]
        key = (ti, ta.spectrum_key, f, att, rel, sh)
        hit = self._smooth.get_(key)
        if hit is not None:
            return hit
        hop_ms = sp.hop / A.SR * 1000
        ca = 1 - np.exp(-hop_ms / att) if att > 0 else 1.0
        cr = 1 - np.exp(-hop_ms / rel) if rel > 0 else 1.0
        tau = max(att, rel)
        M = int(np.ceil(6 * tau / hop_ms)) + 1
        start = max(0, f - M)
        # resume from the newest cached predecessor if present
        for j in range(f - 1, start - 1, -1):
            prev = self._smooth.get((ti, ta.spectrum_key, j, att, rel, sh))
            if prev is not None:
                start = j
                yL, yR = prev[0].copy(), prev[1].copy()
                break
        else:
            fr = self.raw_frame(ta, start)
            yL, yR = fr['L'].copy(), fr['R'].copy()
        for g in range(start + 1, f + 1):
            fr = self.raw_frame(ta, g)
            for y, x in ((yL, fr['L']), (yR, fr['R'])):
                c = np.where(x > y, ca, cr)
                y += c * (x - y)
            if g >= f - 2:
                self._smooth.put((ti, ta.spectrum_key, g, att, rel, sh), (yL.copy(), yR.copy(), fr['ipd']))
        out = (yL, yR, self.raw_frame(ta, f)['ipd'])
        self._smooth.put(key, out)
        return out

    def running_ref(self, ti, ta, release_s):
        key = (ti, ta.spectrum_key, round(release_s, 3))
        arr = self._running.get(key)
        if arr is None:
            fm = ta.spectrum.frame_max.astype(np.float64)
            dec = np.exp(-(ta.spectrum.hop / A.SR) / max(release_s, 1e-3))
            arr = np.empty_like(fm)
            acc = 0.0
            floor = ta.spectrum.ref_global * 0.03
            for i, v in enumerate(fm):
                acc = max(v, acc * dec)
                arr[i] = max(acc, floor)
            self._running[key] = arr
        return arr

    def cumsum(self, ti, ta, name):
        key = (ti, name, ta.key)
        cs = self._cumsum.get(key)
        if cs is None:
            arr = ta.feats[name]
            cs = np.cumsum(arr / (np.percentile(arr, 95) + 1e-9)) / A.FEAT_FPS
            self._cumsum[key] = cs
        return cs

    def ctx(self, ti, f, render, aspect):
        ta = self.tracks()[ti]
        key = (ti, ta.spectrum_key, f, tuple(str(render[k]) for k in CTX_KEYS), round(aspect, 4), self.shifts(ta))
        c = self._ctx.get_(key)
        if c is None:
            c = FrameCtx(self, ti, f, ta, render, aspect)
            self._ctx.put(key, c)
        return c

    # ---------------------------------------------------------------- keyframe resolution
    def _layer(self, k):
        m = MODES[k.mode]
        return Layer(m, k.params, Pal(k.palette.lut(), k.palette.cyclic), k.use_palette, m.name)

    def layers_at(self, t):
        k0, k1, u = self.project.bracket(t)
        if k1 is None or u <= 0:
            return [(1.0, self._layer(k0))], (k0, k1, u)
        if u >= 1:
            return [(1.0, self._layer(k1))], (k0, k1, u)
        if k0.mode == k1.mode and k0.transition == 'morph' and k0.use_palette == k1.use_palette:
            m = MODES[k0.mode]
            params = interpolate_params(m, k0.params, k1.params, u)
            uq = round(u * 256) / 256
            key = (id(k0.palette), id(k1.palette), k0.palette.lut().ctypes.data, k1.palette.lut().ctypes.data, uq)
            lut = self._lut_mix.get_(key)
            if lut is None:
                lut = mix_luts(k0.palette.lut(), k1.palette.lut(), uq)
                self._lut_mix.put(key, lut)
            cyc = k0.palette.cyclic if u < 0.5 else k1.palette.cyclic
            return [(1.0, Layer(m, params, Pal(lut, cyc), k0.use_palette, m.name))], (k0, k1, u)
        return [(1 - u, self._layer(k0)), (u, self._layer(k1))], (k0, k1, u)

    # ---------------------------------------------------------------- dots
    def colourise(self, ctx, layers):
        res = []
        for w, L in layers:
            out = L.mode.fn(ctx, L.params, L.pal, L.use_palette)
            vis = ctx.visible if out.visible is None else out.visible
            g = np.ones(ctx.n, np.float32) if out.gain is None else out.gain
            res.append((w, out, np.where(vis, g, 0.0).astype(np.float32)))
        if len(res) == 1:
            return res[0][1].rgb, res[0][2], res[0][1].overlay
        (w0, o0, g0), (w1, o1, g1) = res
        u = w1
        rgb = mix_oklab(o0.rgb, o1.rgb, u)
        # where only one layer shows a dot, keep that layer's colour
        only0 = (g0 > 0) & (g1 <= 0)
        only1 = (g1 > 0) & (g0 <= 0)
        rgb[only0] = o0.rgb[only0]
        rgb[only1] = o1.rgb[only1]
        gain = (w0 * g0 + w1 * g1).astype(np.float32)
        overlay = (o0.overlay or []) if u < 0.5 else (o1.overlay or [])
        return rgb, gain, overlay

    def frame_dots(self, ti, f, aspect=None):
        aspect = aspect or self.aspect
        ta = self.tracks()[ti]
        rev = (self.project.revision, ta.spectrum_key, ta.status, round(aspect, 4))
        key = (ti, f, rev)
        hit = self._dots.get_(key)
        if hit is not None:
            return hit
        t = self.track_offset(ti) + f * ta.spectrum.hop / A.SR
        render = self.project.render_at(t)
        ctx = self.ctx(ti, f, render, aspect)
        layers, _ = self.layers_at(t)
        if ctx.n == 0:
            out = (np.zeros(0, np.float32),) * 3 + (np.zeros((0, 3), np.float32), [])
        else:
            rgb, gain, overlay = self.colourise(ctx, layers)
            inten = (np.power(np.clip(ctx.level, 0, None), render['gamma']) * render['brightness'] * gain).astype(np.float32)
            keep = inten > 1e-4
            if render['max_dots'] and keep.sum() > render['max_dots']:
                thr = np.partition(inten[keep], -int(render['max_dots']))[-int(render['max_dots'])]
                keep &= inten >= thr
            out = (ctx.x[keep], ctx.y[keep], inten[keep], np.ascontiguousarray(rgb[keep], dtype=np.float32), overlay)
        self._dots.put(key, out)
        return out

    def gather(self, t, aspect=None):
        """All dots visible at timeline time t (with trail). Returns dict for the renderer."""
        aspect = aspect or self.aspect
        with self.lock:
            ti, lt = self.locate(t)
            render = self.project.render_at(t)
            empty = {'x': np.zeros(0, np.float32), 'y': np.zeros(0, np.float32), 'i': np.zeros(0, np.float32),
                     'rgb': np.zeros((0, 3), np.float32), 'overlay': [], 'render': render, 'track': ti, 'local_t': lt}
            if ti is None:
                return empty
            ta = self.tracks()[ti]
            if ta.spectrum is None:
                return empty
            sp = ta.spectrum
            hop_s = sp.hop / A.SR
            f_end = min(int(np.floor(lt / hop_s + 1e-9)), sp.n_frames - 1)
            half = max(render['trail_ms'], 0.0) / 1000.0
            if half <= 0:
                frames = [f_end]
            else:
                K = min(int(np.ceil(half * 6.65 / hop_s)) + 1, 800)
                frames = list(range(max(0, f_end - K + 1), f_end + 1))
            xs, ys, ins, cols = [], [], [], []
            overlay = []
            for f in frames:
                x, y, inten, rgb, ov = self.frame_dots(ti, f, aspect)
                age = lt - f * hop_s
                w = 1.0 if half <= 0 else 0.5 ** (max(age, 0.0) / half)
                xs.append(x)
                ys.append(y)
                ins.append(inten * w)
                cols.append(rgb)
                if f == f_end:
                    overlay = ov
            x, y = np.concatenate(xs), np.concatenate(ys)
            i, rgb = np.concatenate(ins), np.concatenate(cols)
            if render['flip_x']:
                x = -x
            if render['flip_y']:
                y = 1 - y
            if render['mirror_x']:
                x, y, i, rgb = np.concatenate([x, -x]), np.concatenate([y, y]), np.concatenate([i, i]), np.concatenate([rgb, rgb])
            if render['mirror_y']:
                y2 = 0.5 + 0.5 * y
                x, y, i, rgb = np.concatenate([x, x]), np.concatenate([y2, 1 - y2]), np.concatenate([i, i]), np.concatenate([rgb, rgb])
            return {'x': x, 'y': y, 'i': i, 'rgb': rgb, 'overlay': overlay, 'render': render, 'track': ti, 'local_t': lt}

    def invalidate(self):
        with self.lock:
            self._dots.clear()
            self._ctx.clear()
