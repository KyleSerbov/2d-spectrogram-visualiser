"""Offline, deterministic track analysis (GPU via torch/CUDA when available), cached on disk.

Two layers:
* Visual spectrum: depends on the user's Analysis settings (window, hop, window function, zero
  padding, multi-resolution bands). Stored per band as float16 L/R magnitudes (normalised so a
  full-scale sine = 1.0) plus int8 inter-channel phase difference, frames centred at f*hop.
* Features on a fixed grid (2048 / hop 512): onset/flux, spectral centroid, chroma, f0, stereo width,
  RMS, beat tracking, optional Demucs stem ratios and vowel tracking. Independent of the settings.
"""
import hashlib
import json
import math
import os
import queue
import threading
import time
import traceback
from collections import OrderedDict

import numpy as np

from . import audio_io
from .vowel import track_vowels

try:
    import torch
    _HAS_TORCH = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False
    DEVICE = 'cpu'

SR = audio_io.PROJECT_RATE
FEAT_N = 2048
FEAT_HOP = 512
FEAT_FPS = SR / FEAT_HOP
FEATURE_VERSION = 3
SPEC_VERSION = 2
MIN_STORE_FREQ = 10.0

ANALYSIS_DEFAULTS = {
    'window': 2048, 'hop': 512, 'window_fn': 'hann', 'pad': 1,
    'multires': False, 'bass_window': 8192, 'treble_window': 1024,
    'cross_low': 250.0, 'cross_high': 3000.0, 'blend_oct': 0.5,
}
ANALYSIS_PRESETS = {
    'Classic v5': dict(ANALYSIS_DEFAULTS),
    'Punchy beats': dict(ANALYSIS_DEFAULTS, window=1024, hop=256, pad=2),
    'Smooth bass': dict(ANALYSIS_DEFAULTS, window=4096, hop=512, window_fn='blackmanharris', pad=2,
                        multires=True, bass_window=16384, treble_window=2048, cross_low=200.0, cross_high=2000.0),
    'Hi-res detail': dict(ANALYSIS_DEFAULTS, window=8192, hop=512, window_fn='blackmanharris', pad=1,
                          multires=True, bass_window=16384, treble_window=2048, cross_low=300.0, cross_high=4000.0),
}
WINDOW_FUNCS = ['hann', 'blackmanharris', 'gaussian', 'hamming']
WINDOW_SIZES = [512, 1024, 2048, 4096, 8192, 16384]


def cache_root():
    base = os.environ.get('FDS_CACHE') or os.path.join(os.environ.get('LOCALAPPDATA') or os.path.expanduser('~'),
                                                       'fft-dot-studio', 'cache')
    os.makedirs(base, exist_ok=True)
    return base


def file_key(path):
    st = os.stat(path)
    h = hashlib.sha1(f'{os.path.abspath(path).lower()}|{st.st_size}|{int(st.st_mtime)}'.encode()).hexdigest()
    return h[:20]


def settings_key(s):
    s = sanitize_settings(s)
    return hashlib.sha1((json.dumps(s, sort_keys=True) + f'|{SPEC_VERSION}').encode()).hexdigest()[:12]


def sanitize_settings(s):
    out = dict(ANALYSIS_DEFAULTS)
    out.update({k: v for k, v in (s or {}).items() if k in ANALYSIS_DEFAULTS})
    out['window'] = int(out['window'])
    out['bass_window'] = int(out['bass_window'])
    out['treble_window'] = int(out['treble_window'])
    out['pad'] = int(max(1, min(8, out['pad'])))
    wins = [out['window']] + ([out['bass_window'], out['treble_window']] if out['multires'] else [])
    out['hop'] = int(max(32, min(int(out['hop']), min(wins))))
    out['cross_low'] = float(out['cross_low'])
    out['cross_high'] = float(max(out['cross_high'], out['cross_low'] * 1.5))
    out['blend_oct'] = float(max(0.05, min(2.0, out['blend_oct'])))
    out['multires'] = bool(out['multires'])
    if out['window_fn'] not in WINDOW_FUNCS:
        out['window_fn'] = 'hann'
    return out


def get_window(fn, N):
    from scipy.signal import windows
    if fn == 'blackmanharris':
        return windows.blackmanharris(N, sym=True)
    if fn == 'gaussian':
        return windows.gaussian(N, std=N / 6.0, sym=True)
    if fn == 'hamming':
        return np.hamming(N)
    return np.hanning(N)


def band_layout(s):
    """[(N, f_lo, f_hi, blend_lo, blend_hi)] crossfade edges in Hz (None = hard edge)."""
    nyq = SR / 2
    if not s['multires']:
        return [(s['window'], 0.0, nyq, None, None)]
    b = 2 ** (s['blend_oct'] / 2)
    cl, ch = s['cross_low'], s['cross_high']
    return [(s['bass_window'], 0.0, cl * b, None, (cl / b, cl * b)),
            (s['window'], cl / b, ch * b, (cl / b, cl * b), (ch / b, ch * b)),
            (s['treble_window'], ch / b, nyq, (ch / b, ch * b), None)]


def _fade_weights(freqs, lo_blend, hi_blend):
    w = np.ones_like(freqs, dtype=np.float32)
    lf = np.log2(np.maximum(freqs, 1e-3))
    if lo_blend:
        a, b = np.log2(lo_blend[0]), np.log2(lo_blend[1])
        u = np.clip((lf - a) / (b - a), 0, 1)
        w *= (0.5 - 0.5 * np.cos(np.pi * u)).astype(np.float32)
    if hi_blend:
        a, b = np.log2(hi_blend[0]), np.log2(hi_blend[1])
        u = np.clip((lf - a) / (b - a), 0, 1)
        w *= (0.5 + 0.5 * np.cos(np.pi * u)).astype(np.float32)
    return w


# ------------------------------------------------------------------------------------------ STFT

def _frames_count(n, hop):
    return n // hop + 1


def stft_band(x2, N, nfft, hop, win, lo_bin, hi_bin, out_mag, out_ipd, progress=None):
    """x2 float32 (2, n). Writes magnitudes (F, B, 2) float16 and IPD (F, B) int8."""
    n = x2.shape[1]
    Fr = _frames_count(n, hop)
    norm = float(win.sum() / 2.0)
    B = hi_bin - lo_bin
    chunk = int(max(32, min(4096, (1 << 26) // max(nfft, 1))))
    if _HAS_TORCH:
        with torch.no_grad():
            xt = torch.from_numpy(np.ascontiguousarray(x2)).to(DEVICE)
            xt = torch.nn.functional.pad(xt, (N // 2, N // 2 + hop))
            view = xt.unfold(1, N, hop)
            wt = torch.from_numpy(win.astype(np.float32)).to(DEVICE)
            for a in range(0, Fr, chunk):
                b = min(Fr, a + chunk)
                spec = torch.fft.rfft(view[:, a:b] * wt, n=nfft)[..., lo_bin:hi_bin]
                out_mag[a:b] = (spec.abs() / norm).permute(1, 2, 0).to(torch.float16).cpu().numpy()
                ipd = torch.angle(spec[1] * spec[0].conj())
                out_ipd[a:b] = torch.round(ipd / math.pi * 127).to(torch.int8).cpu().numpy()
                if progress:
                    progress(b / Fr)
            del xt, view
    else:
        xp = np.pad(x2, ((0, 0), (N // 2, N // 2 + hop)))
        for a in range(0, Fr, chunk):
            b = min(Fr, a + chunk)
            idx = (np.arange(a, b) * hop)[:, None] + np.arange(N)[None, :]
            spec = np.fft.rfft(xp[:, idx] * win, n=nfft)[..., lo_bin:hi_bin]
            out_mag[a:b] = (np.abs(spec) / norm).transpose(1, 2, 0).astype(np.float16)
            out_ipd[a:b] = np.round(np.angle(spec[1] * np.conj(spec[0])) / np.pi * 127).astype(np.int8)
            if progress:
                progress(b / Fr)
    return Fr, B


class SpecBand:
    def __init__(self, N, nfft, freqs, fade, mag, ipd):
        self.N, self.nfft = N, nfft
        self.freqs, self.fade = freqs, fade
        self.mag, self.ipd = mag, ipd


class Spectrum:
    """Multi-band visual spectrum. get_frame(f) -> dict of concatenated per-dot arrays."""

    def __init__(self, settings, bands, n_frames, frame_max):
        self.settings = settings
        self.hop = settings['hop']
        self.bands = bands
        self.n_frames = n_frames
        self.frame_max = frame_max
        self.freqs = np.concatenate([b.freqs for b in bands]).astype(np.float32)
        self.fade = np.concatenate([b.fade for b in bands]).astype(np.float32)
        self.ref_global = float(np.percentile(frame_max, 99.5)) if len(frame_max) else 1.0
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def frame_time(self, f):
        return f * self.hop / SR

    def frame_at(self, t):
        return int(np.floor(t * SR / self.hop + 1e-9))

    def get_frame_shifted(self, f, shifts):
        """Frame f with each band read at f + shift (fractional frames, linear interpolation)."""
        if not any(shifts):
            return self.get_frame(f)
        key = (f, shifts)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        Ls, Rs, Ps = [], [], []
        for b, s in zip(self.bands, shifts):
            pos = f + s
            a = int(np.floor(pos))
            fr = pos - a
            a0 = min(max(a, 0), self.n_frames - 1)
            a1 = min(max(a + 1, 0), self.n_frames - 1)
            m = b.mag[a0].astype(np.float32) * (1 - fr) + b.mag[a1].astype(np.float32) * fr
            if a + 1 < 0 or a > self.n_frames - 1:
                m = np.zeros_like(m)
            Ls.append(m[:, 0])
            Rs.append(m[:, 1])
            Ps.append(b.ipd[a0 if fr < 0.5 else a1])
        out = {'L': np.concatenate(Ls) * self.fade, 'R': np.concatenate(Rs) * self.fade,
               'ipd': np.concatenate(Ps).astype(np.float32) * (np.pi / 127.0)}
        with self._lock:
            self._cache[key] = out
            if len(self._cache) > 512:
                self._cache.popitem(last=False)
        return out

    def get_frame(self, f):
        with self._lock:
            hit = self._cache.get(f)
            if hit is not None:
                self._cache.move_to_end(f)
                return hit
        mags = [b.mag[f] for b in self.bands]
        ipds = [b.ipd[f] for b in self.bands]
        m = np.concatenate(mags, axis=0).astype(np.float32)
        out = {'L': m[:, 0] * self.fade, 'R': m[:, 1] * self.fade,
               'ipd': np.concatenate(ipds).astype(np.float32) * (np.pi / 127.0)}
        with self._lock:
            self._cache[f] = out
            if len(self._cache) > 512:
                self._cache.popitem(last=False)
        return out


def compute_spectrum(x, settings, cache_dir, progress=None):
    """x float32 (n, 2). Returns Spectrum (memmapped from cache_dir)."""
    s = sanitize_settings(settings)
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, 'meta.json')
    layout = band_layout(s)
    n = x.shape[0]
    Fr = _frames_count(n, s['hop'])
    if os.path.exists(meta_path):
        try:
            return _load_spectrum(cache_dir, s)
        except Exception:
            pass
    x2 = np.ascontiguousarray(x.T.astype(np.float32))
    bands = []
    meta = {'settings': s, 'n_frames': Fr, 'bands': []}
    for bi, (N, f_lo, f_hi, blo, bhi) in enumerate(layout):
        nfft = N * s['pad']
        df = SR / nfft
        lo_bin = max(1, int(np.ceil(max(f_lo, MIN_STORE_FREQ) / df)))
        hi_bin = min(nfft // 2 + 1, int(np.floor(f_hi / df)) + 1)
        if hi_bin <= lo_bin:
            continue
        B = hi_bin - lo_bin
        mag = np.lib.format.open_memmap(os.path.join(cache_dir, f'band{bi}_mag.npy'), mode='w+',
                                        dtype=np.float16, shape=(Fr, B, 2))
        ipd = np.lib.format.open_memmap(os.path.join(cache_dir, f'band{bi}_ipd.npy'), mode='w+',
                                        dtype=np.int8, shape=(Fr, B))
        win = get_window(s['window_fn'], N)
        prog = (lambda p, bi=bi: progress((bi + p) / len(layout))) if progress else None
        stft_band(x2, N, nfft, s['hop'], win, lo_bin, hi_bin, mag, ipd, prog)
        mag.flush()
        ipd.flush()
        freqs = (np.arange(lo_bin, hi_bin) * df).astype(np.float32)
        meta['bands'].append({'index': bi, 'N': N, 'nfft': nfft, 'lo_bin': lo_bin, 'hi_bin': hi_bin,
                              'blend_lo': blo, 'blend_hi': bhi})
        bands.append(SpecBand(N, nfft, freqs, _fade_weights(freqs, blo, bhi), mag, ipd))
    frame_max = _frame_max(bands, Fr)
    np.save(os.path.join(cache_dir, 'frame_max.npy'), frame_max)
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    return Spectrum(s, bands, Fr, frame_max)


def _frame_max(bands, Fr):
    fm = np.zeros(Fr, np.float32)
    for b in bands:
        for a in range(0, Fr, 2048):
            blk = b.mag[a:a + 2048].astype(np.float32).max(axis=2) * b.fade[None, :]
            fm[a:a + 2048] = np.maximum(fm[a:a + 2048], blk.max(axis=1))
    return fm


def _load_spectrum(cache_dir, s):
    with open(os.path.join(cache_dir, 'meta.json')) as f:
        meta = json.load(f)
    bands = []
    for bm in meta['bands']:
        bi = bm['index']
        mag = np.load(os.path.join(cache_dir, f'band{bi}_mag.npy'), mmap_mode='r')
        ipd = np.load(os.path.join(cache_dir, f'band{bi}_ipd.npy'), mmap_mode='r')
        df = SR / bm['nfft']
        freqs = (np.arange(bm['lo_bin'], bm['hi_bin']) * df).astype(np.float32)
        blo = tuple(bm['blend_lo']) if bm['blend_lo'] else None
        bhi = tuple(bm['blend_hi']) if bm['blend_hi'] else None
        bands.append(SpecBand(bm['N'], bm['nfft'], freqs, _fade_weights(freqs, blo, bhi), mag, ipd))
    frame_max = np.load(os.path.join(cache_dir, 'frame_max.npy'))
    return Spectrum(s, bands, meta['n_frames'], frame_max)


# -------------------------------------------------------------------------------------- features

def _feature_stft(x):
    """Complex STFT on the feature grid for L and R: returns torch/numpy complex (2, F, 1025)."""
    win = np.hanning(FEAT_N).astype(np.float32)
    x2 = np.ascontiguousarray(x.T.astype(np.float32))
    n = x2.shape[1]
    Fr = _frames_count(n, FEAT_HOP)
    if _HAS_TORCH:
        with torch.no_grad():
            xt = torch.nn.functional.pad(torch.from_numpy(x2).to(DEVICE), (FEAT_N // 2, FEAT_N // 2 + FEAT_HOP))
            view = xt.unfold(1, FEAT_N, FEAT_HOP)[:, :Fr]
            wt = torch.from_numpy(win).to(DEVICE)
            out = []
            for a in range(0, Fr, 4096):
                out.append(torch.fft.rfft(view[:, a:a + 4096] * wt).cpu())
            spec = torch.cat(out, dim=1).numpy()
    else:
        xp = np.pad(x2, ((0, 0), (FEAT_N // 2, FEAT_N // 2 + FEAT_HOP)))
        spec = np.empty((2, Fr, FEAT_N // 2 + 1), np.complex64)
        for a in range(0, Fr, 2048):
            b = min(Fr, a + 2048)
            idx = (np.arange(a, b) * FEAT_HOP)[:, None] + np.arange(FEAT_N)[None, :]
            spec[:, a:b] = np.fft.rfft(xp[:, idx] * win)
    return spec, win.sum() / 2.0


def feature_freqs():
    return np.fft.rfftfreq(FEAT_N, 1.0 / SR).astype(np.float32)


def compute_features(x, spec, norm):
    """x (n,2) float32; spec complex (2, F, 1025). Returns dict of per-frame feature arrays."""
    freqs = feature_freqs()
    magL = np.abs(spec[0]).astype(np.float32) / norm
    magR = np.abs(spec[1]).astype(np.float32) / norm
    M = 0.5 * (magL + magR)
    F = M.shape[0]
    feats = {}
    # RMS (time domain, on the feature grid)
    mono = x.mean(axis=1)
    n = len(mono)
    centers = np.arange(F) * FEAT_HOP
    sq = np.concatenate([[0.0], np.cumsum(mono.astype(np.float64) ** 2)])
    a = np.clip(centers - FEAT_N // 2, 0, n)
    b = np.clip(centers + FEAT_N // 2, 0, n)
    feats['rms'] = np.sqrt((sq[b] - sq[a]) / np.maximum(b - a, 1)).astype(np.float32)
    # Spectral flux onsets (total + 3 bands)
    logM = np.log1p(1000.0 * M)
    flux = np.maximum(0.0, np.diff(logM, axis=0, prepend=logM[:1]))
    bands = [(20, 250), (250, 4000), (4000, SR / 2)]
    ob = np.stack([flux[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) for lo, hi in bands], axis=1)
    onset = flux[:, freqs >= 20].sum(axis=1)

    def nrm(v):
        p = np.percentile(v, 99.5) if len(v) else 1.0
        return np.clip(v / (p + 1e-9), 0, 2).astype(np.float32)
    feats['onset'] = nrm(onset)
    feats['onset_bands'] = np.stack([nrm(ob[:, i]) for i in range(3)], axis=1)
    # Centroid
    sel = freqs >= 20
    feats['centroid'] = ((M[:, sel] * freqs[sel]).sum(1) / (M[:, sel].sum(1) + 1e-9)).astype(np.float32)
    # Chroma (soft semitone assignment, 55 Hz - 5 kHz)
    with np.errstate(divide='ignore'):
        midi = 69 + 12 * np.log2(np.maximum(freqs, 1e-3) / 440.0)
    W = np.zeros((len(freqs), 12), np.float32)
    ok = (freqs >= 55) & (freqs <= 5000)
    lo = np.floor(midi).astype(int) % 12
    fr = (midi - np.floor(midi)).astype(np.float32)
    W[ok, lo[ok]] += 1 - fr[ok]
    W[ok, (lo[ok] + 1) % 12] += fr[ok]
    chroma = (M ** 2) @ W
    feats['chroma'] = (chroma / (chroma.max(axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    # Stereo width
    feats['width'] = (np.abs(magL - magR).sum(1) / ((magL + magR).sum(1) + 1e-9)).astype(np.float32)
    # Fundamental (harmonic sum with sub-harmonic suppression)
    feats['f0'], feats['f0_conf'] = _f0_harmonic_sum(M)
    # Beats
    env = _smooth(feats['onset'], 2)
    beats, tempo = beat_track(env, FEAT_FPS)
    feats['beats'] = beats.astype(np.float32)
    feats['tempo'] = np.array([tempo], np.float32)
    feats['loud_ref'] = np.array([float(np.percentile(M.max(axis=1), 99.5)) if F else 1.0], np.float32)
    return feats


def _smooth(v, sigma):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(v.astype(np.float64), sigma, mode='nearest')


def _f0_harmonic_sum(M):
    df = SR / FEAT_N
    cands = 45.0 * 2 ** (np.arange(0, int(48 * np.log2(1000 / 45.0))) / 48.0)
    K = 10
    k = np.arange(1, K + 1)
    pos = (cands[:, None] * k[None, :]) / df
    posh = (cands[:, None] * (k[None, :] - 0.5)) / df
    wk = (1.0 / k ** 0.6)[None, :]
    nb = M.shape[1]

    def interp(Mx, p):
        p = np.clip(p, 0, nb - 2)
        i = np.floor(p).astype(int)
        f = (p - i).astype(np.float32)
        return Mx[:, i] * (1 - f) + Mx[:, i + 1] * f
    out_f0 = np.zeros(M.shape[0], np.float32)
    out_c = np.zeros(M.shape[0], np.float32)
    Ms = np.sqrt(M)
    for a in range(0, M.shape[0], 1024):
        blk = Ms[a:a + 1024]
        if _HAS_TORCH and DEVICE == 'cuda':
            with torch.no_grad():
                bt = torch.from_numpy(np.ascontiguousarray(blk)).to(DEVICE)

                def tint(p):
                    p = np.clip(p, 0, nb - 2)
                    i = torch.from_numpy(np.floor(p).astype(np.int64)).to(DEVICE)
                    f = torch.from_numpy((p - np.floor(p)).astype(np.float32)).to(DEVICE)
                    return bt[:, i] * (1 - f) + bt[:, i + 1] * f
                wkt = torch.from_numpy(wk.astype(np.float32)).to(DEVICE)
                score = ((tint(pos) - 0.5 * tint(posh)) * wkt).sum(-1)
                best = torch.argmax(score, dim=1)
                sc = score.gather(1, best[:, None])[:, 0]
                conf = sc / (score.clamp(min=0).mean(dim=1) + 1e-9)
                out_f0[a:a + 1024] = torch.from_numpy(cands.astype(np.float32)).to(DEVICE)[best].cpu().numpy()
                out_c[a:a + 1024] = conf.cpu().numpy()
        else:
            score = ((interp(blk, pos) - 0.5 * interp(blk, posh)) * wk).sum(-1)
            best = score.argmax(axis=1)
            out_f0[a:a + 1024] = cands[best]
            out_c[a:a + 1024] = score[np.arange(len(best)), best] / (np.maximum(score, 0).mean(1) + 1e-9)
    return out_f0, np.clip(out_c / 8.0, 0, 1).astype(np.float32)


def beat_track(env, fps, tightness=100.0):
    """Ellis (2007) dynamic-programming beat tracker. Returns (beat times s, tempo BPM)."""
    o = env - env.mean()
    o = o / (o.std() + 1e-9)
    n = len(o)
    if n < fps * 2:
        return np.zeros(0), 120.0
    maxlag = int(fps * 60 / 50)
    seg = o[:min(n, int(fps * 90))]
    ac = np.correlate(seg, seg, mode='full')[len(seg) - 1:][:maxlag]
    lags = np.arange(1, len(ac))
    bpm = 60 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpm / 120.0) / 0.9) ** 2)
    score = ac[1:] * prior * ((bpm >= 50) & (bpm <= 220))
    P = int(lags[np.argmax(score)])
    half = P // 2
    if half >= 3 and ac[half - 1:half + 2].max() >= 0.45 * ac[P] and 80 <= 60 * fps / half <= 190:
        P = half - 1 + int(np.argmax(ac[half - 1:half + 2]))
    tempo = 60 * fps / P
    C = o.copy()
    back = np.full(n, -1)
    lo, hi = int(round(P / 2)), int(round(2 * P))
    offs = np.arange(lo, hi + 1)
    pen = -tightness * np.log(offs / P) ** 2
    for t in range(lo, n):
        prev = t - offs
        valid = prev >= 0
        if not valid.any():
            continue
        cand = np.where(valid, C[np.maximum(prev, 0)] + pen, -np.inf)
        j = int(np.argmax(cand))
        if cand[j] > 0:
            C[t] = o[t] + cand[j]
            back[t] = prev[j]
    tail = max(0, n - P)
    t = tail + int(np.argmax(C[tail:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    beats = np.array(beats[::-1], dtype=np.float64) / fps
    return beats, float(tempo)


# --------------------------------------------------------------------------------------- stems

_DEMUCS = {'model': None, 'error': None}
_GPU_LOCK = threading.Lock()


def demucs_available():
    if _DEMUCS['error']:
        return False
    try:
        import demucs  # noqa: F401
        return _HAS_TORCH
    except Exception:
        return False


def separate_stems(x, progress=None):
    """Demucs htdemucs on GPU. x (n,2) float32 @44.1k -> dict name -> (n,2) float32."""
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    if _DEMUCS['model'] is None:
        m = get_model('htdemucs')
        m.eval()
        _DEMUCS['model'] = m
    model = _DEMUCS['model']
    with torch.no_grad():
        wav = torch.from_numpy(np.ascontiguousarray(x.T)).float()
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav = (wav - mean) / std
        out = apply_model(model, wav[None], device=DEVICE, split=True, overlap=0.25, progress=False)[0]
        out = out * std + mean
    return {name: out[i].T.numpy().astype(np.float32) for i, name in enumerate(model.sources)}


STEM_ORDER = ['vocals', 'drums', 'bass', 'other']


def stem_ratios(stems, n_frames):
    """Per feature-frame, per-bin share of each stem's magnitude -> uint8 (F, 1025, 4)."""
    mags = []
    for name in STEM_ORDER:
        sp, norm = _feature_stft(stems[name])
        mags.append((np.abs(sp[0]) + np.abs(sp[1]))[:n_frames].astype(np.float32))
    tot = sum(mags) + 1e-6
    out = np.stack([np.round(m / tot * 255) for m in mags], axis=-1).astype(np.uint8)
    return out


def centre_vocal_estimate(spec, n):
    """Fallback vocal isolation without Demucs: centre-panned, harmonic, 120 Hz-6 kHz content."""
    from scipy.ndimage import median_filter
    L, R = spec[0], spec[1]
    aL, aR = np.abs(L), np.abs(R)
    centre = np.clip(1 - np.abs(aL - aR) / (aL + aR + 1e-9), 0, 1) ** 4
    mid = 0.5 * (L + R)
    mag = np.abs(mid)
    harm = median_filter(mag, size=(9, 1))
    perc = median_filter(mag, size=(1, 9))
    hmask = harm ** 2 / (harm ** 2 + perc ** 2 + 1e-12)
    freqs = feature_freqs()
    band = ((freqs > 120) & (freqs < 6000)).astype(np.float32)
    est = mid * centre * hmask * band[None, :]
    win = np.hanning(FEAT_N)
    frames = np.fft.irfft(est, FEAT_N, axis=1) * win
    out = np.zeros(n + FEAT_N * 2)
    wsum = np.zeros_like(out)
    for f in range(frames.shape[0]):
        s = f * FEAT_HOP
        out[s:s + FEAT_N] += frames[f]
        wsum[s:s + FEAT_N] += win ** 2
    out = out / np.maximum(wsum, 1e-6)
    return out[FEAT_N // 2:FEAT_N // 2 + n].astype(np.float32)


# ------------------------------------------------------------------------------- track analysis

class TrackAnalysis:
    """Everything known about one audio file. Fields fill in as the background worker progresses."""

    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.status = 'queued'
        self.progress = 0.0
        self.error = None
        self.info = None
        self.audio = None           # int16 (n, 2) for playback
        self.n_samples = 0
        self.duration = 0.0
        self.envelope = None
        self.spectrum = None
        self.spectrum_key = None
        self.feats = None
        self.stems = None           # uint8 (F, 1025, 4) memmap: vocals drums bass other
        self.vowel = None
        self.vocal_source = None
        self.key = None
        self._float_audio = None
        self.lock = threading.Lock()

    @property
    def dir(self):
        return os.path.join(cache_root(), self.key)

    def feat_index(self, t):
        if self.feats is None:
            return 0
        n = len(self.feats['onset'])
        return int(min(max(round(t * FEAT_FPS), 0), n - 1))

    def float_audio(self):
        if self._float_audio is not None:
            return self._float_audio
        return self.audio.astype(np.float32) / 32767.0 if self.audio is not None else None


class AnalysisManager:
    """Single background worker so GPU jobs never overlap. Jobs: load, spectrum, features, stems, vowels."""

    def __init__(self, use_demucs=True, sync=False, vocal_source='auto'):
        self.use_demucs = use_demucs
        self.vocal_source = vocal_source
        self.sync = sync
        self.q = queue.PriorityQueue()
        self.counter = 0
        self.tracks = {}
        self.settings = dict(ANALYSIS_DEFAULTS)
        self.listeners = []
        self.revision = 0
        self.thread = None
        if not sync:
            self.thread = threading.Thread(target=self._run, daemon=True, name='fds-analysis')
            self.thread.start()

    def _push(self, prio, job, track):
        self.counter += 1
        if self.sync:
            self._do(job, track)
        else:
            self.q.put((prio, self.counter, job, track))

    def get(self, path):
        ta = self.tracks.get(path)
        if ta is None:
            ta = TrackAnalysis(path)
            self.tracks[path] = ta
            self._push(0, 'load', ta)
        return ta

    def set_settings(self, settings):
        s = sanitize_settings(settings)
        if s == sanitize_settings(self.settings):
            return
        self.settings = s
        for ta in self.tracks.values():
            if ta.audio is not None:
                self._push(1, 'spectrum', ta)

    def redo_vocals(self):
        for ta in self.tracks.values():
            if ta.feats is not None:
                ta.vowel = None
                ta.stems = None
                ta.status = 'stems / vowels'
                self._push(3, 'stems', ta)

    def busy(self):
        return (not self.q.empty()) or any(t.status not in ('ready', 'error') for t in self.tracks.values())

    def _run(self):
        while True:
            _prio, _c, job, ta = self.q.get()
            self._do(job, ta)

    def _do(self, job, ta):
        try:
            if job == 'load':
                self._load(ta)
            elif job == 'spectrum':
                self._spectrum(ta)
            elif job == 'features':
                self._features(ta)
            elif job == 'stems':
                self._stems(ta)
        except Exception as e:
            ta.error = f'{job}: {e}'
            ta.status = 'error'
            traceback.print_exc()
        self.revision += 1

    def _load(self, ta):
        ta.status = 'decoding'
        ta.progress = 0.0
        ta.key = file_key(ta.path)
        x, info = audio_io.load_audio(ta.path)
        ta.info = info
        ta._float_audio = x
        ta.audio = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
        ta.n_samples = len(x)
        ta.duration = len(x) / SR
        ta.envelope = audio_io.envelope(x, SR)
        os.makedirs(ta.dir, exist_ok=True)
        self._spectrum(ta)
        self._push(2, 'features', ta)

    def _spectrum(self, ta):
        s = sanitize_settings(self.settings)
        key = settings_key(s)
        if ta.spectrum_key == key:
            return
        ta.status = 'spectrum'
        x = ta.float_audio()

        def prog(p):
            ta.progress = p
        with _GPU_LOCK:
            spec = compute_spectrum(x, s, os.path.join(ta.dir, 'spec_' + key), prog)
        ta.spectrum, ta.spectrum_key = spec, key
        if ta.status == 'spectrum':
            ta.status = 'features' if ta.feats is None else 'ready'

    def _features(self, ta):
        path = os.path.join(ta.dir, f'features_v{FEATURE_VERSION}.npz')
        x = ta.float_audio()
        if os.path.exists(path):
            d = np.load(path)
            ta.feats = {k: d[k] for k in d.files}
        else:
            ta.status = 'features'
            with _GPU_LOCK:
                spec, norm = _feature_stft(x)
            ta.feats = compute_features(x, spec, norm)
            np.savez(path, **ta.feats)
        ta.status = 'stems / vowels'
        self._push(3, 'stems', ta)

    def _stems(self, ta):
        x = ta.float_audio()
        n_feat = len(ta.feats['onset'])
        stems_path = os.path.join(ta.dir, f'stems_v{FEATURE_VERSION}.npy')
        src_pref = self.vocal_source if self.use_demucs else 'centre'
        vowel_path = os.path.join(ta.dir, f'vowel_v{FEATURE_VERSION}_{src_pref}.npz')
        vocal = None
        if self.use_demucs and demucs_available():
            if os.path.exists(stems_path) and os.path.exists(vowel_path):
                ta.stems = np.load(stems_path, mmap_mode='r')
            else:
                ta.status = 'separating stems (Demucs)'
                try:
                    with _GPU_LOCK:
                        stems = separate_stems(x)
                        ratios = stem_ratios(stems, n_feat)
                    np.save(stems_path, ratios)
                    ta.stems = np.load(stems_path, mmap_mode='r')
                    vocal = stems['vocals'].mean(axis=1)
                    np.save(os.path.join(ta.dir, 'vocal_rms.npy'), np.array([np.sqrt(np.mean(vocal ** 2))]))
                except Exception as e:
                    _DEMUCS['error'] = str(e)
                    traceback.print_exc()
            ta.vocal_source = 'demucs' if ta.stems is not None else 'centre'
        if os.path.exists(vowel_path):
            d = np.load(vowel_path)
            ta.vowel = {k: d[k] for k in d.files}
        else:
            ta.status = 'vowel tracking'
            centre = None
            if vocal is None or src_pref in ('auto', 'centre'):
                spec, _ = _feature_stft(x)
                centre = centre_vocal_estimate(spec, len(x))
            if vocal is not None and src_pref == 'auto' and centre is not None:
                # Demucs only knows real voices: if its vocal stem is near-silent compared with the
                # centre-channel estimate (synthetic/instrumental leads), use the centre estimate instead.
                if np.sqrt(np.mean(vocal ** 2)) < 0.25 * np.sqrt(np.mean(centre ** 2)):
                    vocal = None
            if vocal is None or src_pref == 'centre':
                vocal = centre
                ta.vocal_source = 'centre'
            times = np.arange(n_feat) / FEAT_FPS
            ta.vowel = track_vowels(vocal, SR, times)
            np.savez(vowel_path, **ta.vowel)
        ta._float_audio = None
        ta.status = 'ready'
        ta.progress = 1.0
