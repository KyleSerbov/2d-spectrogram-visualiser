"""Synthetic test music: stereo drums, bass, panned chords and a sung vowel line (for --selftest)."""
import numpy as np

from .vowel import VOWELS, synth_vowel

SR = 44100


def _env(n, a, d):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / d)


def kick(n):
    t = np.arange(n) / SR
    f = 50 + 120 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.002, 0.18)


def snare(n, rng):
    return (rng.standard_normal(n) * 0.6 + np.sin(2 * np.pi * 190 * np.arange(n) / SR)) * _env(n, 0.001, 0.09)


def hat(n, rng):
    x = rng.standard_normal(n)
    x = np.diff(x, prepend=0)
    return x * _env(n, 0.0005, 0.025) * 0.3


def song(duration=12.0, bpm=120.0, seed=1, vowel_seq=None):
    """Returns (stereo float32 (n,2), dict of ground truth: beat times, vowel segments)."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    beat = 60.0 / bpm
    beats = np.arange(0, duration - 0.3, beat)
    for i, bt in enumerate(beats):
        s = int(bt * SR)
        k = kick(min(int(0.4 * SR), n - s))
        L[s:s + len(k)] += 0.8 * k
        R[s:s + len(k)] += 0.8 * k
        if i % 2 == 1:
            sn = snare(min(int(0.25 * SR), n - s), rng)
            L[s:s + len(sn)] += 0.35 * sn
            R[s:s + len(sn)] += 0.35 * sn
        for h in (0.0, 0.5):
            hs = int((bt + h * beat) * SR)
            if hs < n:
                hh = hat(min(int(0.08 * SR), n - hs), rng)
                L[hs:hs + len(hh)] += 0.15 * hh
                R[hs:hs + len(hh)] += 0.45 * hh
    t = np.arange(n) / SR
    roots = [45, 45, 41, 43]  # A2 A2 F2 G2 (midi)
    bar = 4 * beat
    for bi in range(int(duration / bar) + 1):
        r = roots[bi % 4]
        s, e = int(bi * bar * SR), min(n, int((bi + 1) * bar * SR))
        if s >= n:
            break
        tt = t[s:e] - t[s]
        fb = 440 * 2 ** ((r - 12 - 69) / 12)
        bass = sum(np.sin(2 * np.pi * fb * h * tt) / h for h in range(1, 9))
        L[s:e] += 0.35 * bass * np.minimum(1, tt / 0.01)
        R[s:e] += 0.35 * bass * np.minimum(1, tt / 0.01)
        for j, iv in enumerate((0, 4, 7, 12)):
            fc = 440 * 2 ** ((r + 12 + iv - 69) / 12)
            pan = (-0.8, -0.3, 0.3, 0.8)[j]
            tone = sum(np.sin(2 * np.pi * fc * h * tt + h) / h ** 1.2 for h in range(1, 12)) * np.exp(-tt / 1.6) * 0.08
            L[s:e] += tone * (1 - pan) / 2 * 2
            R[s:e] += tone * (1 + pan) / 2 * 2
    pad = rng.standard_normal(n)
    from scipy.signal import lfilter
    pad = lfilter([0.02], [1, -0.98], pad)
    L += 0.05 * pad
    R += 0.05 * np.roll(pad, 300)
    seq = vowel_seq or VOWELS * 4
    seg = 0.5
    vtimes = []
    for i, v in enumerate(seq):
        st = 1.0 + i * seg
        if st + seg > duration:
            break
        f0 = 440 * 2 ** ((57 + [0, 3, 5, 7, 10][i % 5] - 69) / 12)
        y = synth_vowel(v, f0=f0, dur=seg, vibrato=0.01, seed=i) * 0.5
        s = int(st * SR)
        L[s:s + len(y)] += y
        R[s:s + len(y)] += y
        vtimes.append((st, st + seg, v))
    x = np.stack([L, R], axis=1)
    x = x / (np.abs(x).max() + 1e-9) * 0.9
    return x.astype(np.float32), {'beats': beats, 'vowels': vtimes, 'bpm': bpm}
