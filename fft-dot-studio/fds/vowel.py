"""Vowel / mouth-shape tracking from an (isolated) vocal signal.

Two complementary estimators run on 40 ms frames at 11.025 kHz:

* Vowel weights (what drives the colours): analysis-by-synthesis. The pitch is tracked, the spectrum is
  sampled at each harmonic, and each vowel's formant-resonator template is fitted to those harmonic
  amplitudes after removing level + spectral tilt. This stays reliable for high sung pitches, where
  classic LPC formant picking breaks down. A per-track formant-scale search normalises for the singer
  (child / female / male vocal-tract length).
* F1/F2 (for the vowel chart): LPC (batched Levinson) -> pole frequencies via batched companion-matrix
  eigenvalues -> Viterbi tracking of the (F1, F2) pair with continuity costs.
"""
from math import gcd

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import lfilter, resample_poly

VOWELS = ['A', 'E', 'I', 'O', 'U']
VOWEL_F1 = np.array([750.0, 470.0, 300.0, 500.0, 330.0])
VOWEL_F2 = np.array([1250.0, 1950.0, 2300.0, 900.0, 800.0])
DEFAULT_VOWEL_COLORS = {
    'A': [1.00, 0.20, 0.18],   # wide open: hot red
    'E': [1.00, 0.80, 0.10],   # spread: yellow
    'I': [0.20, 1.00, 0.45],   # smile: green
    'O': [0.25, 0.45, 1.00],   # round: blue
    'U': [0.75, 0.25, 1.00],   # pursed: violet
}
FS = 11025
LPC_ORDER = 12
FRAME_S = 0.040

REF_FORMANTS = np.array([[750, 1250, 2500, 3400],    # A
                         [470, 1950, 2600, 3400],    # E
                         [300, 2300, 3000, 3600],    # I
                         [500, 900, 2500, 3400],     # O
                         [330, 800, 2400, 3300]],    # U
                        dtype=np.float64)
REF_BW = np.array([80.0, 100.0, 150.0, 200.0])
ALPHA_GRID = np.array([0.84, 0.90, 0.96, 1.0, 1.05, 1.11, 1.18, 1.26])
HARM_FMAX = 4000.0
HARM_K = 40


def bark(f):
    f = np.asarray(f, dtype=np.float64)
    return 26.81 * f / (1960.0 + f) - 0.53


def lpc_batch(frames, order):
    """frames (F, N) -> LPC coefficients a (F, order+1) with a[:,0]=1 (autocorrelation method)."""
    F, N = frames.shape
    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    spec = np.fft.rfft(frames, nfft, axis=1)
    r = np.fft.irfft(np.abs(spec) ** 2, nfft, axis=1)[:, :order + 1]
    r[:, 0] = r[:, 0] * (1.0 + 1e-9) + 1e-12
    lag = np.arange(order + 1)
    r *= np.exp(-0.5 * (2 * np.pi * 40.0 * lag / FS) ** 2)
    a = np.zeros((F, order + 1))
    a[:, 0] = 1.0
    err = r[:, 0].copy()
    for i in range(1, order + 1):
        acc = r[:, i].copy()
        if i > 1:
            acc += np.einsum('fj,fj->f', a[:, 1:i], r[:, i - 1:0:-1])
        k = -acc / np.maximum(err, 1e-15)
        a_prev = a.copy()
        a[:, i] = k
        if i > 1:
            a[:, 1:i] = a_prev[:, 1:i] + k[:, None] * a_prev[:, i - 1:0:-1]
        err *= (1.0 - k * k)
    return a


def lpc_roots(a):
    """Batched polynomial roots via companion-matrix eigenvalues. a (F, p+1) -> (F, p) complex."""
    F, P1 = a.shape
    p = P1 - 1
    comp = np.zeros((F, p, p))
    comp[:, 0, :] = -a[:, 1:]
    if p > 1:
        comp[:, np.arange(1, p), np.arange(0, p - 1)] = 1.0
    return np.linalg.eigvals(comp)


def formant_candidates(roots, max_cands=6):
    ang = np.angle(roots)
    freq = ang * FS / (2 * np.pi)
    bw = -FS / np.pi * np.log(np.maximum(np.abs(roots), 1e-9))
    ok = (ang > 0) & (freq > 180) & (freq < 4000) & (bw < 700)
    freq = np.where(ok, freq, np.inf)
    order = np.argsort(freq, axis=1)[:, :max_cands]
    return np.take_along_axis(freq, order, axis=1), np.take_along_axis(np.where(ok, bw, np.inf), order, axis=1)


def viterbi_formants(cf, cb, voiced):
    """Choose one (F1, F2) pair per frame from LPC candidates with continuity costs."""
    F, M = cf.shape
    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
    pi = np.array([p[0] for p in pairs])
    pj = np.array([p[1] for p in pairs])
    f1, f2 = cf[:, pi], cf[:, pj]
    b1, b2 = cb[:, pi], cb[:, pj]
    with np.errstate(invalid='ignore'):
        valid = (np.isfinite(f1) & np.isfinite(f2) & (f1 > 200) & (f1 < 1200) & (f2 > 550) & (f2 < 3200)
                 & (f2 - f1 > 180))
    z1, z2 = bark(np.where(valid, f1, 500)), bark(np.where(valid, f2, 1500))
    local = np.where(valid, b1, 1e3) / 400.0 + np.where(valid, b2, 1e3) / 600.0
    local += 0.08 * ((z1 - bark(500)) ** 2 + 0.5 * (z2 - bark(1500)) ** 2) + 0.15 * pi
    local = np.where(valid, local, 1e6)
    P = len(pairs)
    cost = local[0].copy()
    back = np.zeros((F, P), dtype=np.int32)
    w = np.where(voiced, 1.0, 0.2)
    for t in range(1, F):
        d = (z1[t][None, :] - z1[t - 1][:, None]) ** 2 + 0.6 * (z2[t][None, :] - z2[t - 1][:, None]) ** 2
        tot = cost[:, None] + w[t] * 2.5 * d
        back[t] = np.argmin(tot, axis=0)
        cost = tot[back[t], np.arange(P)] + local[t]
    path = np.zeros(F, dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for t in range(F - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    idx = np.arange(F)
    ok = valid[idx, path]
    return np.where(ok, f1[idx, path], np.nan), np.where(ok, f2[idx, path], np.nan)


def _autocorr(raw):
    F, N = raw.shape
    x = raw - raw.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    s = np.fft.rfft(x * np.hanning(N), nfft, axis=1)
    ac = np.fft.irfft(np.abs(s) ** 2, nfft, axis=1)[:, :N]
    wac = np.fft.irfft(np.abs(np.fft.rfft(np.hanning(N), nfft)) ** 2, nfft)[:N]
    return ac / (ac[:, :1] + 1e-12) / np.maximum(wac / wac[0], 0.05)


def pitch_track(raw):
    """Normalised-autocorrelation pitch per frame (70-1000 Hz), avoiding sub-octave errors.
    Returns (f0 Hz, periodicity 0..1)."""
    F, N = raw.shape
    ac = _autocorr(raw)
    lo, hi = int(FS / 1000), min(int(FS / 70), N - 2)
    seg = ac[:, lo:hi]
    mx = seg.max(axis=1)
    is_peak = (seg[:, 1:-1] >= seg[:, :-2]) & (seg[:, 1:-1] >= seg[:, 2:]) & (seg[:, 1:-1] >= 0.9 * mx[:, None])
    first = np.where(is_peak.any(axis=1), is_peak.argmax(axis=1) + 1, seg.argmax(axis=1))
    lag = first + lo
    ii = np.arange(F)
    lm, l0, lp = ac[ii, np.clip(lag - 1, 0, N - 1)], ac[ii, lag], ac[ii, np.clip(lag + 1, 0, N - 1)]
    den = lm - 2 * l0 + lp
    delta = np.where(np.abs(den) > 1e-9, 0.5 * (lm - lp) / np.where(np.abs(den) > 1e-9, den, 1.0), 0.0)
    f0 = FS / (lag + np.clip(delta, -0.5, 0.5))
    return f0, np.clip(mx, 0, 1)


def voicing(periodicity, energy_db):
    periodic = np.clip((periodicity - 0.35) / 0.35, 0, 1)
    loud = np.clip((energy_db + 42.0) / 12.0, 0, 1)
    return periodic * loud


def harmonic_amplitudes(frames_w, f0, nfft=4096):
    """Peak dB magnitude near each harmonic k*f0. Returns (H (F,K) dB, freqs (F,K), valid (F,K))."""
    F = frames_w.shape[0]
    spec = 20 * np.log10(np.abs(np.fft.rfft(frames_w, nfft, axis=1)) + 1e-9)
    df = FS / nfft
    k = np.arange(1, HARM_K + 1)[None, :]
    fk = f0[:, None] * k
    valid = fk < HARM_FMAX
    halfw = np.maximum(1, np.round(0.25 * f0 / df)).astype(np.int64)
    W = int(halfw.max()) if F else 1
    offs = np.arange(-W, W + 1)
    center = np.round(fk / df).astype(np.int64)
    H = np.full((F, HARM_K), -200.0)
    for c0 in range(0, F, 512):
        sl = slice(c0, c0 + 512)
        idx = np.clip(center[sl][:, :, None] + offs[None, None, :], 0, spec.shape[1] - 1)
        rows = np.arange(idx.shape[0])[:, None, None]
        vals = spec[sl][rows, idx]
        inwin = np.abs(offs)[None, None, :] <= halfw[sl][:, None, None]
        H[sl] = np.where(inwin, vals, -200.0).max(axis=2)
    top = np.where(valid, H, -200).max(axis=1, keepdims=True)
    valid &= H > top - 45.0
    return H, fk, valid


def resonator_db(f, formants, bws):
    """Magnitude (dB) of a cascade of unity-DC-gain 2-pole resonators at frequencies f."""
    out = np.zeros(np.shape(f), dtype=np.float64)
    z1 = np.exp(-1j * 2 * np.pi * np.asarray(f) / FS)
    for fc, bw in zip(formants, bws):
        r = np.exp(-np.pi * bw / FS)
        c = 2 * r * np.cos(2 * np.pi * fc / FS)
        out += 20 * np.log10(np.abs((1 - c + r * r) / (1 - c * z1 + r * r * z1 * z1)) + 1e-12)
    return out


def template_distances(H, fk, valid):
    """RMS dB misfit (F, V, A) between harmonic amplitudes and each vowel template at each formant
    scale, after removing the best-fitting level + spectral tilt (weighted least squares)."""
    F = H.shape[0]
    V, A = len(REF_FORMANTS), len(ALPHA_GRID)
    D = np.full((F, V, A), 99.0)
    m = valid.astype(np.float64)
    n = m.sum(axis=1)
    lf = np.log2(np.maximum(fk, 1.0))
    s1, sx, sxx = n, (m * lf).sum(1), (m * lf * lf).sum(1)
    det = s1 * sxx - sx * sx
    safe_det = np.where(det > 1e-9, det, 1.0)
    for v in range(V):
        for a in range(A):
            r = (H - resonator_db(fk, REF_FORMANTS[v] * ALPHA_GRID[a], REF_BW)) * m
            sy, sxy = r.sum(1), (r * lf).sum(1)
            c1 = np.where(det > 1e-9, (s1 * sxy - sx * sy) / safe_det, 0.0)
            c0 = (sy - c1 * sx) / np.maximum(s1, 1)
            res = (r - (c0[:, None] + c1[:, None] * lf) * m)
            D[:, v, a] = np.sqrt((res ** 2).sum(1) / np.maximum(n, 1))
    D[n < 3] = 99.0
    return D


def track_vowels(vocal, sr, frame_times, normalise='shift', smooth_ms=40.0, temperature=2.0):
    """vocal: mono float array at sr; frame_times: seconds. Returns dict of per-frame arrays:
    weights (F,5) A E I O U, voiced 0..1, openness 0..1, f0, f1, f2 (smoothed), alpha (singer scale)."""
    x = np.asarray(vocal, dtype=np.float64)
    if sr != FS:
        g = gcd(int(sr), FS)
        x = resample_poly(x, FS // g, int(sr) // g)
    xe = lfilter([1.0, -0.94], [1.0], x)
    N = int(FRAME_S * FS)
    centers = np.round(np.asarray(frame_times) * FS).astype(np.int64)
    xp = np.pad(xe, (N, N))
    xr = np.pad(x, (N, N))
    starts = np.clip(centers - N // 2 + N, 0, len(xp) - N)
    idx = starts[:, None] + np.arange(N)[None, :]
    frames = xp[idx] * np.hamming(N)[None, :]
    raw = xr[idx]
    rms = np.sqrt((raw ** 2).mean(axis=1) + 1e-12)
    ref = np.percentile(rms, 99.0) + 1e-9
    energy_db = 20 * np.log10(rms / ref + 1e-9)
    Fn = len(centers)
    f1 = np.full(Fn, np.nan)
    f2 = np.full(Fn, np.nan)
    f0 = np.full(Fn, np.nan)
    period = np.zeros(Fn)
    D = np.full((Fn, len(REF_FORMANTS), len(ALPHA_GRID)), 99.0)
    active = energy_db > -60
    if active.any():
        p0, per = pitch_track(raw[active])
        f0[active], period[active] = p0, per
        vp = voicing(per, energy_db[active])
        a = lpc_batch(frames[active], LPC_ORDER)
        cf, cb = formant_candidates(lpc_roots(a))
        f1[active], f2[active] = viterbi_formants(cf, cb, vp > 0.5)
        H, fk, valid = harmonic_amplitudes(raw[active] * np.hanning(N)[None, :], p0)
        D[active] = template_distances(H, fk, valid)
    voiced_p = voicing(period, energy_db) * (D.min(axis=(1, 2)) < 50)
    hop = float(np.median(np.diff(frame_times))) if len(frame_times) > 1 else 0.01
    good = np.isfinite(f1) & np.isfinite(f2)

    def smooth(v):
        if good.sum() < 2:
            return v
        lv = np.log(np.where(good, v, 1.0))
        ii = np.arange(len(lv))
        lv = np.interp(ii, ii[good], lv[good])
        lv = median_filter(lv, size=5, mode='nearest')
        return np.exp(gaussian_filter1d(lv, max(smooth_ms / 1000.0 / hop, 0.5), mode='nearest'))
    f1s, f2s = smooth(f1), smooth(f2)
    voiced = gaussian_filter1d(voiced_p, 1.5, mode='nearest')
    vmask = voiced > 0.5
    a_idx = int(np.argmin(np.abs(ALPHA_GRID - 1.0)))
    if normalise == 'shift' and vmask.sum() > 60:
        a_idx = int(np.argmin(np.median(D[vmask].min(axis=1), axis=0)))
    Dv = D.min(axis=2) if normalise == 'frame' else D[:, :, a_idx]
    logits = -(Dv ** 2) / (2 * temperature ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)
    sig = max(smooth_ms / 1000.0 / hop, 0.5)
    vw = np.maximum(voiced, 1e-3)[:, None]
    w = gaussian_filter1d(w * vw, sig, axis=0, mode='nearest') / gaussian_filter1d(vw, sig, axis=0, mode='nearest')
    w /= w.sum(axis=1, keepdims=True)
    z1w = w @ bark(REF_FORMANTS[:, 0])
    openness = np.clip((z1w - bark(280)) / (bark(800) - bark(280)), 0, 1)
    fit = float(np.median(Dv[vmask].min(axis=1))) if vmask.any() else 99.0
    return {'f1': f1s.astype(np.float32), 'f2': f2s.astype(np.float32), 'voiced': voiced.astype(np.float32),
            'weights': w.astype(np.float32), 'openness': openness.astype(np.float32),
            'f0': np.nan_to_num(f0).astype(np.float32), 'alpha': np.array([ALPHA_GRID[a_idx]], np.float32),
            'quality': np.array([vmask.mean(), fit], np.float32)}


# ---------------------------------------------------------------- synthesis (tests / demos)

def synth_vowel(vowel, f0=140.0, dur=0.6, sr=44100, vibrato=0.0, seed=0, formant_scale=1.0, jitter=0.0):
    """Source-filter vowel: glottal pulse train through cascade formant resonators.
    Uses formant values that deliberately differ from the classifier's templates (F3/F4, bandwidths)."""
    rng = np.random.default_rng(seed)
    k = VOWELS.index(vowel)
    F = [VOWEL_F1[k], VOWEL_F2[k], 2600.0 + 200 * (k == 2), 3500.0]
    F = [f * formant_scale * (1 + jitter * rng.uniform(-1, 1)) for f in F]
    B = [70.0, 110.0, 140.0, 220.0]
    n = int(dur * sr)
    t = np.arange(n) / sr
    f0t = f0 * (1 + vibrato * np.sin(2 * np.pi * 5.5 * t))
    phase = np.cumsum(f0t) / sr
    src = np.zeros(n)
    src[np.nonzero(np.diff(np.floor(phase)) > 0)[0]] = 1.0
    src = lfilter([1.0], [1.0, -1.9, 0.9025], src)
    src += 0.002 * rng.standard_normal(n)
    y = src
    for fc, bw in zip(F, B):
        r = np.exp(-np.pi * bw / sr)
        th = 2 * np.pi * fc / sr
        y = lfilter([1 - 2 * r * np.cos(th) + r * r], [1, -2 * r * np.cos(th), r * r], y)
    env = np.minimum(1, np.minimum(t / 0.03, (dur - t) / 0.03))
    y = y * env
    return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)


def synth_vowel_sequence(vowels, f0=160.0, seg=0.6, sr=44100, **kw):
    return np.concatenate([synth_vowel(v, f0=f0, dur=seg, sr=sr, seed=i, **kw) for i, v in enumerate(vowels)])
