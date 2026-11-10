"""Colour modes. Each mode maps the dots of one analysis frame to colours.

A mode returns ModeOut(rgb, gain, visible, overlay):
  rgb      (N,3) sRGB 0..1
  gain     (N,) brightness multiplier or None (=1)
  visible  (N,) bool override of the default threshold visibility, or None
Most modes compute a scalar s that a palette maps to colour. Modes with intrinsic colours
(note hues, vowels, stems...) return their native colours unless the layer's "use palette" is on,
in which case their scalar is sent through the palette instead.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .colorspace import (adjust_lightness, desaturate, hsv_to_rgb, mix_oklab, oklab_to_srgb,
                         srgb_to_oklab, weighted_oklab)
from .vowel import DEFAULT_VOWEL_COLORS, VOWELS


# ------------------------------------------------------------------------------------ framework

@dataclass
class Param:
    key: str
    label: str
    kind: str                  # float int bool choice color zones
    default: object
    lo: float = 0.0
    hi: float = 1.0
    choices: list = field(default_factory=list)
    tip: str = ''
    log: bool = False


@dataclass
class ModeOut:
    rgb: np.ndarray
    gain: np.ndarray = None
    visible: np.ndarray = None
    overlay: list = None


class Mode:
    def __init__(self, key, name, category, desc, params, fn, palette='Rainbow (HSV)', native=False, needs=()):
        self.key, self.name, self.category, self.desc = key, name, category, desc
        self.params, self.fn = params, fn
        self.palette, self.native, self.needs = palette, native, set(needs)

    def defaults(self):
        return {p.key: (list(p.default) if isinstance(p.default, list) else p.default) for p in self.params}

    def fill(self, params):
        d = self.defaults()
        if params:
            d.update({k: v for k, v in params.items() if k in d})
        return d


MODES = {}
MODE_ORDER = []


def mode(key, name, category, desc, params=(), palette='Rainbow (HSV)', native=False, needs=()):
    def deco(fn):
        MODES[key] = Mode(key, name, category, desc, list(params), fn, palette, native, needs)
        MODE_ORDER.append(key)
        return fn
    return deco


class Pal:
    """Palette sampler passed to modes (possibly a blend of two keyframes' palettes)."""

    def __init__(self, lut, cyclic):
        self.lut, self.cyclic = lut, cyclic

    def __call__(self, s):
        s = np.asarray(s, dtype=np.float32)
        n = self.lut.shape[0]
        if self.cyclic:
            idx = np.floor((s % 1.0) * n).astype(np.int64) % n
        else:
            idx = np.clip(np.round(np.nan_to_num(s) * (n - 1)), 0, n - 1).astype(np.int64)
        return self.lut[idx]


def pick(native_rgb, s, pal, use_palette):
    if use_palette or native_rgb is None:
        return pal(s)
    return native_rgb


# ------------------------------------------------------------------------------------ helpers

def midi_of(f):
    return 69.0 + 12.0 * np.log2(np.maximum(f, 1e-3) / 440.0)


def rainbow_lerp(midi, order='chromatic'):
    """v6 'note rainbow': HSV hue of the semitone below and above, blended linearly in RGB."""
    base = np.floor(midi).astype(np.int64) % 12
    frac = (midi - np.floor(midi)).astype(np.float32)[:, None]
    nxt = (base + 1) % 12
    if order == 'fifths':
        base, nxt = (base * 7) % 12, (nxt * 7) % 12
    return hsv_to_rgb(base / 12.0) * (1 - frac) + hsv_to_rgb(nxt / 12.0) * frac


def note_hue(midi):
    return hsv_to_rgb((midi % 12.0) / 12.0)


def greedy_peaks(x, y, amp, cand, min_dist, limit=700):
    """Original greedy selection: strongest first, accept if >= min_dist from every accepted peak."""
    idx = np.nonzero(cand)[0]
    if len(idx) == 0:
        return idx
    idx = idx[np.argsort(-amp[idx], kind='stable')][:limit]
    pts = np.stack([x[idx], y[idx]], axis=1)
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    supp = np.zeros(len(idx), bool)
    chosen = []
    for i in range(len(idx)):
        if supp[i]:
            continue
        chosen.append(i)
        supp |= D[i] < min_dist
    return idx[np.array(chosen, dtype=np.int64)]


def nearest(xa, ya, xb, yb):
    d = (xa[:, None] - xb[None, :]) ** 2 + (ya[:, None] - yb[None, :]) ** 2
    return np.argmin(d, axis=1), np.sqrt(d)


def band_peaks(ctx, amps, prominence, height):
    """scipy find_peaks per contiguous analysis band (bins are frequency-ordered inside a band)."""
    out = []
    for a, b in ctx.band_slices:
        seg = amps[a:b]
        if len(seg) < 3:
            continue
        pk, _ = find_peaks(seg, height=height, prominence=prominence)
        out.append(pk + a)
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def parabolic_freq(ctx, amps, idx):
    f = ctx.freq[idx].copy()
    ok = (idx > 0) & (idx < len(amps) - 1)
    i = idx[ok]
    m1, m0, p1 = amps[i - 1], amps[i], amps[i + 1]
    den = m1 - 2 * m0 + p1
    d = np.where(np.abs(den) > 1e-12, 0.5 * (m1 - p1) / np.where(np.abs(den) > 1e-12, den, 1), 0)
    d = np.where(np.abs(d) <= 0.5, d, 0)
    df = np.abs(ctx.freq[np.minimum(i + 1, len(amps) - 1)] - ctx.freq[i])
    f[ok] = ctx.freq[i] + d * df
    return f


def screen_xy(ctx):
    """Dot positions in height units (x spans the aspect ratio)."""
    return (ctx.x + 1) * 0.5 * ctx.aspect, ctx.y


def centre_of(ctx, p):
    X, Y = screen_xy(ctx)
    c = p.get('centre', 'screen')
    if c == 'centroid':
        w = ctx.level * ctx.visible
        if w.sum() > 1e-9:
            return float((X * w).sum() / w.sum()), float((Y * w).sum() / w.sum())
        return ctx.aspect / 2, 0.5
    if c == 'custom':
        return (p.get('cx', 0.0) + 1) * 0.5 * ctx.aspect, p.get('cy', 0.5)
    return ctx.aspect / 2, 0.5


def curve(u, kind, g=1.0):
    u = np.clip(u, 0, 1)
    if kind == 'smooth':
        return u * u * (3 - 2 * u)
    if kind == 'exponential':
        return 1 - np.exp(-4 * g * u) if g > 0 else u
    if kind == 'gamma':
        return u ** max(g, 1e-3)
    if kind == 'sine':
        return 0.5 - 0.5 * np.cos(np.pi * u)
    if kind == 'steps':
        n = max(2, int(g * 4))
        return np.floor(u * n) / (n - 1)
    return u


CENTRE_PARAMS = [
    Param('centre', 'Centre', 'choice', 'screen', choices=['screen', 'centroid', 'custom'],
          tip='Screen centre, the live amplitude-weighted centroid of the dots, or a custom point.'),
    Param('cx', 'Custom X', 'float', 0.0, -1, 1, tip='Custom centre X (-1 left .. 1 right).'),
    Param('cy', 'Custom Y', 'float', 0.5, 0, 1, tip='Custom centre Y (0 bottom .. 1 top).'),
]


def _vowel_hold(ta):
    """Per-track cached 'last voiced vowel' weights (hold through consonants/rests)."""
    cache = getattr(ta, '_vowel_hold', None)
    if cache is not None:
        return cache
    w = ta.vowel['weights']
    v = ta.vowel['voiced']
    held = w.copy()
    last = np.full(w.shape[1], 1.0 / w.shape[1], np.float32)
    for i in range(len(w)):
        if v[i] > 0.5:
            last = w[i]
        else:
            held[i] = last
    ta._vowel_hold = held
    return held


# ============================================================================ PORTED MODES

@mode('mono', 'Mono White (v5)', 'Ported', 'The original v1-v5 look: every dot white, brightness from amplitude.',
      palette='White')
def m_mono(ctx, p, pal, up):
    rgb = np.ones((ctx.n, 3), np.float32)
    return ModeOut(pick(rgb, np.full(ctx.n, 0.5), pal, up))


@mode('note_rainbow', 'Note Rainbow (v6 rainbow)', 'Ported',
      'Every dot coloured by its own pitch: semitone hues (C red ... B magenta) blended linearly in RGB '
      'between the nearest semitones. Faithful port of v6-colour-rainbow.',
      [Param('order', 'Note order', 'choice', 'chromatic', choices=['chromatic', 'fifths'],
             tip='Chromatic = hue wheel by semitone (original). Fifths = circle of fifths, so consonant notes get neighbouring hues.')],
      native=True)
def m_note_rainbow(ctx, p, pal, up):
    midi = midi_of(ctx.freq)
    s = (midi % 12) / 12 if p['order'] == 'chromatic' else ((np.floor(midi) * 7) % 12 + (midi % 1)) / 12
    return ModeOut(pick(rainbow_lerp(midi, p['order']), s, pal, up))


@mode('note_hue', 'Note Hue Continuous (portfolio)', 'Ported',
      'Continuous hue = (MIDI pitch mod 12)/12 for every dot (the get_note_color of the portfolio version).',
      native=True)
def m_note_hue(ctx, p, pal, up):
    midi = midi_of(ctx.freq)
    s = (midi % 12) / 12
    return ModeOut(pick(note_hue(midi), s, pal, up))


@mode('note_regions', 'Note Regions (v6 notes / v7)', 'Ported',
      'Greedy peak picking in (pan, log-freq) space; every dot takes the semitone colour of its nearest peak, '
      'giving solid colour regions. Port of v6-colour-notes and v7-beat-detection.',
      [Param('min_dist', 'Min peak distance', 'float', 0.15, 0.005, 0.5, tip='Minimum distance between peaks in normalised pan/frequency space.'),
       Param('min_amp', 'Min peak amplitude', 'float', 0.5, 0.0, 1.0, tip='Peaks must be at least this fraction of the loudest dot.')],
      native=True)
def m_note_regions(ctx, p, pal, up):
    amp = ctx.amp_rel
    cand = ctx.visible & (amp >= p['min_amp'] * (amp[ctx.visible].max() if ctx.visible.any() else 1))
    pk = greedy_peaks(ctx.x, ctx.y, amp, cand, p['min_dist'])
    if len(pk) == 0:
        return ModeOut(pick(np.ones((ctx.n, 3), np.float32), np.full(ctx.n, 0.5), pal, up))
    midi = midi_of(ctx.freq[pk])
    pc = rainbow_lerp(midi)
    near, _ = nearest(ctx.x, ctx.y, ctx.x[pk], ctx.y[pk])
    return ModeOut(pick(pc[near], ((midi % 12) / 12)[near], pal, up))


@mode('note_regions_blend', 'Note Regions Blend (v6 notes-blending)', 'Ported',
      'find_peaks with parabolic frequency refinement, hue-interpolated peak colours, and inverse-distance '
      'weighted blending of all peak colours for every dot. Port of v6-colour-notes-blending.',
      [Param('min_dist', 'Min peak distance', 'float', 0.15, 0.005, 0.5),
       Param('min_amp', 'Min peak amplitude', 'float', 0.5, 0.0, 1.0),
       Param('power', 'Blend power', 'float', 1.0, 0.2, 6.0, tip='Higher = sharper boundaries between peak colours.'),
       Param('eps', 'Blend epsilon', 'float', 0.01, 0.001, 0.2)],
      native=True)
def m_note_regions_blend(ctx, p, pal, up):
    amp = ctx.amp_rel
    pk = band_peaks(ctx, np.where(ctx.visible, amp, 0), None, ctx.threshold)
    if len(pk) == 0:
        return ModeOut(pick(np.ones((ctx.n, 3), np.float32), np.full(ctx.n, 0.5), pal, up))
    tf = parabolic_freq(ctx, amp, pk)
    keep = amp[pk] >= p['min_amp']
    pk, tf = pk[keep], tf[keep]
    if len(pk) == 0:
        return ModeOut(pick(np.ones((ctx.n, 3), np.float32), np.full(ctx.n, 0.5), pal, up))
    sel = greedy_peaks(ctx.x[pk], ctx.y[pk], amp[pk], np.ones(len(pk), bool), p['min_dist'])
    pk, tf = pk[sel], tf[sel]
    midi = midi_of(tf)
    base = np.floor(midi) % 12
    frac = midi - np.floor(midi)
    hue = (base / 12.0) * (1 - frac) + ((base + 1) / 12.0) * frac
    cols = hsv_to_rgb(hue)
    _, d = nearest(ctx.x, ctx.y, ctx.x[pk], ctx.y[pk])
    w = 1.0 / (d + p['eps']) ** p['power']
    rgb = (w @ cols) / w.sum(axis=1, keepdims=True)
    s = (w @ hue) / w.sum(axis=1)
    return ModeOut(pick(rgb.astype(np.float32), s, pal, up))


@mode('peaks_rainbow', 'Peaks Only Rainbow (rainbow-improved)', 'Ported',
      'Only spectral peaks (scipy find_peaks with prominence, separate bass/treble bands) are drawn, each in its '
      'rainbow semitone colour, brightness relative to its band. Port of v6-colour-rainbow-improved.',
      [Param('crossover', 'Crossover Hz', 'float', 1000.0, 100, 8000, log=True),
       Param('prominence', 'Prominence', 'float', 0.05, 0.0, 0.5, tip='Peak prominence as a fraction of the band maximum.')],
      native=True)
def m_peaks_rainbow(ctx, p, pal, up):
    vis = np.zeros(ctx.n, bool)
    gain = np.ones(ctx.n, np.float32)
    tf = ctx.freq.copy()
    for lo_band in (True, False):
        bm = (ctx.freq < p['crossover']) if lo_band else (ctx.freq >= p['crossover'])
        if not bm.any():
            continue
        a = np.where(bm, ctx.amp, 0)
        mx = a.max() + 1e-12
        pk = band_peaks(ctx, a, p['prominence'] * mx, ctx.threshold * mx)
        vis[pk] = True
        gain[bm] = ctx.frame_max / mx
        tf[pk] = parabolic_freq(ctx, a, pk)
    midi = midi_of(tf)
    return ModeOut(pick(rainbow_lerp(midi), (midi % 12) / 12, pal, up), gain=gain, visible=vis)


@mode('peak_halo', 'Peak Halo (rainbow-improved-blending)', 'Ported',
      'Peaks plus every dot within a frequency window around them (down to a very low threshold), all in the '
      "peak's semitone colour; quieter bass band. Port of v6-colour-rainbow-improved-blending.",
      [Param('crossover', 'Crossover Hz', 'float', 1000.0, 100, 8000, log=True),
       Param('prominence', 'Prominence', 'float', 0.05, 0.0, 0.5),
       Param('window_hz', 'Halo window Hz', 'float', 50.0, 5, 400, log=True),
       Param('surround', 'Halo threshold', 'float', 0.001, 0.0001, 0.05, log=True),
       Param('low_gain', 'Bass gain', 'float', 0.125, 0.0, 1.0, tip='Original used brightness 1 for bass vs 8 for treble.')],
      native=True)
def m_peak_halo(ctx, p, pal, up):
    vis = np.zeros(ctx.n, bool)
    gain = np.ones(ctx.n, np.float32)
    rgb = np.ones((ctx.n, 3), np.float32)
    s = np.zeros(ctx.n, np.float32)
    for lo_band in (True, False):
        bm = (ctx.freq < p['crossover']) if lo_band else (ctx.freq >= p['crossover'])
        if not bm.any():
            continue
        a = np.where(bm, ctx.amp, 0)
        mx = a.max() + 1e-12
        pk = band_peaks(ctx, a, p['prominence'] * mx, ctx.threshold * mx)
        if len(pk) == 0:
            continue
        cand = np.nonzero(bm & (a > p['surround'] * mx))[0]
        dfreq = np.abs(ctx.freq[cand][:, None] - ctx.freq[pk][None, :])
        near = dfreq.argmin(axis=1)
        ok = dfreq.min(axis=1) < p['window_hz']
        cand, near = cand[ok], near[ok]
        midi = midi_of(ctx.freq[pk])
        rgb[cand] = rainbow_lerp(midi)[near]
        s[cand] = ((midi % 12) / 12)[near]
        vis[cand] = True
        gain[bm] = (ctx.frame_max / mx) * (p['low_gain'] if lo_band else 1.0)
    return ModeOut(pick(rgb, s, pal, up), gain=gain, visible=vis)


@mode('split_rainbow_regions', 'Split: Bass Rainbow / Treble Regions (blending-update)', 'Ported',
      'Bass dots coloured individually by pitch; treble dots take the colour of their nearest greedy peak. '
      'Port of v6-colour-rainbow-improved-blending-update.',
      [Param('crossover', 'Crossover Hz', 'float', 1000.0, 100, 8000, log=True),
       Param('min_dist', 'Min peak distance', 'float', 0.15, 0.005, 0.5),
       Param('min_amp', 'Min peak amplitude', 'float', 0.5, 0.0, 1.0),
       Param('low_gain', 'Bass gain', 'float', 0.125, 0.0, 1.0)],
      native=True)
def m_split(ctx, p, pal, up):
    midi = midi_of(ctx.freq)
    rgb = rainbow_lerp(midi)
    s = (midi % 12) / 12
    gain = np.ones(ctx.n, np.float32)
    lo = ctx.freq < p['crossover']
    hi = ~lo
    vis = ctx.visible.copy()
    for bm in (lo, hi):
        if bm.any():
            mx = ctx.amp[bm].max() + 1e-12
            vis[bm] = ctx.amp[bm] > ctx.threshold * mx
            gain[bm] = ctx.frame_max / mx
    gain[lo] *= p['low_gain']
    if hi.any():
        rel = np.where(hi, ctx.amp / (ctx.amp[hi].max() + 1e-12), 0)
        cand = hi & vis & (rel >= p['min_amp'])
        pk = greedy_peaks(ctx.x, ctx.y, rel, cand, p['min_dist'])
        hi_idx = np.nonzero(hi)[0]
        if len(pk):
            near, _ = nearest(ctx.x[hi_idx], ctx.y[hi_idx], ctx.x[pk], ctx.y[pk])
            rgb[hi_idx] = rgb[pk][near]
            s[hi_idx] = s[pk][near]
        else:
            rgb[hi_idx] = 1.0
    return ModeOut(pick(rgb, s, pal, up), gain=gain, visible=vis)


@mode('peak_sparkle', 'Peak Sparkle (portfolio v1 / update-2)', 'Ported',
      'Bass and treble handled separately; only greedy peaks (tiny min distance) are drawn, in continuous note '
      'hue, brightness relative to the band. Port of v6-...-update-2 (the first GitHub portfolio version).',
      [Param('crossover', 'Crossover Hz', 'float', 1000.0, 100, 8000, log=True),
       Param('min_dist', 'Min peak distance', 'float', 0.01, 0.001, 0.3),
       Param('min_amp', 'Min peak amplitude', 'float', 0.05, 0.0, 1.0)],
      native=True)
def m_sparkle(ctx, p, pal, up):
    vis = np.zeros(ctx.n, bool)
    gain = np.ones(ctx.n, np.float32)
    for bm in (ctx.freq < p['crossover'], ctx.freq >= p['crossover']):
        if not bm.any():
            continue
        mx = ctx.amp[bm].max() + 1e-12
        rel = np.where(bm, ctx.amp / mx, 0)
        cand = bm & (rel > ctx.threshold) & (rel >= p['min_amp'])
        pk = greedy_peaks(ctx.x, ctx.y, rel, cand, p['min_dist'])
        vis[pk] = True
        gain[bm] = ctx.frame_max / mx
    midi = midi_of(ctx.freq)
    return ModeOut(pick(note_hue(midi), (midi % 12) / 12, pal, up), gain=gain, visible=vis)


def _zone_counts(ctx, zones):
    X, Y = screen_xy(ctx)
    out = []
    for z in zones:
        zx = (z['x'] + 1) * 0.5 * ctx.aspect
        inside = ((X - zx) ** 2 + (Y - z['y']) ** 2) < z['r'] ** 2
        out.append((inside, int((inside & ctx.visible).sum())))
    return out


@mode('beat_zones', 'Beat Zones (v7)', 'Ported',
      'Circular detector zones (v7 beat detection): when more dots than the threshold fall inside a zone it '
      'triggers and flashes. Dots inside a flashing zone take its colour. Place zones from the inspector or '
      'by clicking the viewer while "Place zone" is armed.',
      [Param('zones', 'Zones', 'zones', [{'x': -0.45, 'y': 0.25, 'r': 0.08, 'threshold': 12, 'color': [1.0, 0.25, 0.3]},
                                       {'x': 0.45, 'y': 0.25, 'r': 0.08, 'threshold': 12, 'color': [0.3, 0.6, 1.0]}]),
       Param('base', 'Base colouring', 'choice', 'white', choices=['white', 'note regions', 'note rainbow']),
       Param('flash_ms', 'Flash decay ms', 'float', 70.0, 10, 1000, log=True),
       Param('overlay', 'Draw zone rings', 'bool', True),
       Param('boost', 'Flash brightness boost', 'float', 1.0, 0, 4)],
      palette='White')
def m_beat_zones(ctx, p, pal, up):
    if p['base'] == 'note regions':
        rgb = m_note_regions(ctx, MODES['note_regions'].defaults(), pal, False).rgb.copy()
    elif p['base'] == 'note rainbow':
        rgb = rainbow_lerp(midi_of(ctx.freq))
    else:
        rgb = pal(np.full(ctx.n, 0.5)) if up else np.ones((ctx.n, 3), np.float32)
    gain = np.ones(ctx.n, np.float32)
    zones = p['zones'] or []
    overlay = []
    if zones:
        cur = _zone_counts(ctx, zones)
        flash = np.zeros(len(zones))
        steps = max(1, int(np.ceil(p['flash_ms'] / 1000.0 / ctx.hop_s)))
        for j in range(steps + 1):
            prev = ctx.prev(j)
            if prev is None:
                break
            counts = cur if j == 0 else _zone_counts(prev, zones)
            decay = max(0.0, 1.0 - j * ctx.hop_s / (p['flash_ms'] / 1000.0))
            for zi, (_inside, c) in enumerate(counts):
                if c > zones[zi]['threshold']:
                    flash[zi] = max(flash[zi], decay)
        for zi, (inside, c) in enumerate(cur):
            if flash[zi] > 0:
                col = np.array(zones[zi]['color'], np.float32)
                rgb[inside] = mix_oklab(rgb[inside], np.broadcast_to(col, rgb[inside].shape), flash[zi])
                gain[inside] *= 1 + p['boost'] * flash[zi]
            if p['overlay']:
                overlay.append(('ring', zones[zi]['x'], zones[zi]['y'], zones[zi]['r'],
                                [float(v) for v in zones[zi]['color']], float(max(flash[zi], 0.25))))
    return ModeOut(rgb, gain=gain, overlay=overlay)


# ============================================================================ NEW MODES

@mode('radial', 'Radial Glow', 'New',
      'One colour at the centre shifting through the palette as distance from the centre grows. Centre can be '
      'the screen centre, the live centroid of the dots (it follows the music), or a custom point.',
      CENTRE_PARAMS + [
          Param('radius', 'Radius', 'float', 0.6, 0.05, 2.0, tip='Distance (in screen heights) that reaches the end of the palette.'),
          Param('curve', 'Falloff curve', 'choice', 'smooth', choices=['linear', 'smooth', 'exponential', 'gamma', 'sine', 'steps']),
          Param('curve_amount', 'Curve amount', 'float', 1.0, 0.1, 5.0, tip='Gamma exponent / exponential rate / number of steps.'),
          Param('invert', 'Invert', 'bool', False),
          Param('pulse', 'Beat pulse', 'float', 0.0, 0.0, 1.0, tip='Radius swells on every detected beat.')],
      palette='Radial Glow')
def m_radial(ctx, p, pal, up):
    X, Y = screen_xy(ctx)
    cx, cy = centre_of(ctx, p)
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    rad = p['radius'] * (1 + p['pulse'] * 0.5 * ctx.beat_flash(0.15))
    s = curve(r / max(rad, 1e-3), p['curve'], p['curve_amount'])
    if p['invert']:
        s = 1 - s
    return ModeOut(pal(s))


@mode('polar', 'Polar Swirl', 'New',
      'Colour by angle around a centre; the wheel rotates over time (or per beat) and can twist into a spiral.',
      CENTRE_PARAMS + [
          Param('speed', 'Rotation (turns/s)', 'float', 0.1, -2, 2),
          Param('per_beat', 'Rotate per beat instead', 'bool', False),
          Param('twist', 'Spiral twist', 'float', 0.0, -4, 4, tip='Adds radius-dependent rotation.'),
          Param('arms', 'Repeats', 'int', 1, 1, 8)],
      palette='Rainbow (HSV)')
def m_polar(ctx, p, pal, up):
    X, Y = screen_xy(ctx)
    cx, cy = centre_of(ctx, p)
    ang = np.arctan2(Y - cy, X - cx) / (2 * np.pi)
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    rot = ctx.beat_pos() * p['speed'] if p['per_beat'] else ctx.t * p['speed']
    return ModeOut(pal((ang * p['arms'] + rot + p['twist'] * r) % 1.0))


@mode('amp_heat', 'Amplitude Heat', 'New', 'Loud dots glow hot, quiet dots cool, through the palette.',
      [Param('gamma', 'Gamma', 'float', 0.6, 0.1, 3.0),
       Param('db', 'Use dB', 'bool', True), Param('floor', 'dB floor', 'float', -50, -100, -10)],
      palette='Heat')
def m_amp_heat(ctx, p, pal, up):
    a = ctx.amp_rel
    s = np.clip((20 * np.log10(np.maximum(a, 1e-9)) - p['floor']) / -p['floor'], 0, 1) if p['db'] else a
    return ModeOut(pal(s ** p['gamma']))


@mode('freq_gradient', 'Frequency Gradient', 'New', 'Colour by height (frequency): bass to treble through the palette.',
      [Param('offset', 'Offset', 'float', 0.0, -1, 1), Param('repeat', 'Repeats', 'float', 1.0, 0.25, 8.0),
       Param('scroll', 'Scroll speed', 'float', 0.0, -2, 2, tip='Palette scroll per second.')],
      palette='Turbo')
def m_freq_gradient(ctx, p, pal, up):
    s = ctx.y * p['repeat'] + p['offset'] + ctx.t * p['scroll']
    if pal.cyclic or p['repeat'] != 1 or p['scroll'] != 0:
        s = s % 1.0
    return ModeOut(pal(s))


@mode('stereo', 'Stereo Field', 'New', 'Colour by pan position: left / centre / right, or by stereo width.',
      [Param('absolute', 'Width only (|pan|)', 'bool', False), Param('contrast', 'Contrast', 'float', 1.0, 0.2, 4.0)],
      palette='Stereo (L blue / R red)')
def m_stereo(ctx, p, pal, up):
    x = np.clip(ctx.pan * p['contrast'], -1, 1)
    s = np.abs(x) if p['absolute'] else (x + 1) / 2
    return ModeOut(pal(s))


@mode('harmonic', 'Harmonic Series', 'New',
      'Detects the fundamental of each moment and colours dots by which harmonic they are (1st, 2nd, 3rd...). '
      'Inharmonic dots (between harmonics) fade to grey, revealing noise vs. tone.',
      [Param('reference', 'Fundamental', 'choice', 'detected f0', choices=['detected f0', 'strongest peak']),
       Param('max_harm', 'Harmonics spanned', 'int', 12, 2, 32),
       Param('inharmonic_grey', 'Inharmonic greying', 'float', 1.0, 0, 1)],
      palette='Plasma')
def m_harmonic(ctx, p, pal, up):
    if p['reference'] == 'strongest peak' or ctx.feat is None:
        f0 = float(ctx.freq[int(np.argmax(np.where(ctx.freq > 40, ctx.amp, 0)))])
    else:
        f0 = float(ctx.feat_val('f0'))
    n = ctx.freq / max(f0, 1.0)
    k = np.round(n)
    dev = np.abs(n - k) * 2
    s = np.clip((k - 1) / max(p['max_harm'] - 1, 1), 0, 1)
    rgb = desaturate(pal(s), p['inharmonic_grey'] * np.clip(dev * 1.5, 0, 1))
    rgb[k < 1] *= 0.35
    return ModeOut(rgb)


@mode('onset_flash', 'Onset Flash', 'New',
      'Base colour by frequency; every transient (per bass / mid / treble band) flashes the matching dots '
      'towards the flash colour and boosts them. Great for drums.',
      [Param('flash_color', 'Flash colour', 'color', [1.0, 1.0, 1.0]),
       Param('amount', 'Flash amount', 'float', 0.9, 0, 1), Param('decay_ms', 'Decay ms', 'float', 120.0, 10, 1000, log=True),
       Param('boost', 'Brightness boost', 'float', 1.5, 0, 6), Param('per_band', 'Per band', 'bool', True),
       Param('sensitivity', 'Sensitivity', 'float', 1.0, 0.2, 4)],
      palette='Ocean')
def m_onset_flash(ctx, p, pal, up):
    base = pal(ctx.y)
    if ctx.feat is None:
        return ModeOut(base)
    fl = ctx.onset_decay(p['decay_ms'] / 1000.0, bands=p['per_band']) * p['sensitivity']
    if p['per_band']:
        b = np.where(ctx.freq < 250, 0, np.where(ctx.freq < 4000, 1, 2))
        f = np.clip(fl[b] - 0.25, 0, 1.5)
    else:
        f = np.full(ctx.n, np.clip(fl - 0.25, 0, 1.5))
    col = np.broadcast_to(np.array(p['flash_color'], np.float32), base.shape)
    rgb = mix_oklab(base, col, np.clip(f * p['amount'], 0, 1))
    return ModeOut(rgb, gain=(1 + p['boost'] * f).astype(np.float32))


@mode('timbre', 'Timbre (Spectral Centroid)', 'New',
      'The whole picture shifts colour with the brightness of the sound (spectral centroid): dull/bassy '
      'passages at one end of the palette, bright/airy ones at the other.',
      [Param('spread', 'Per-dot spread', 'float', 0.3, 0, 1), Param('lo_hz', 'Dull Hz', 'float', 300, 50, 3000, log=True),
       Param('hi_hz', 'Bright Hz', 'float', 6000, 1000, 16000, log=True)],
      palette='Sunset')
def m_timbre(ctx, p, pal, up):
    c = ctx.feat_val('centroid', smooth=4) if ctx.feat is not None else 1500
    sc = np.clip(np.log(max(c, 1) / p['lo_hz']) / np.log(p['hi_hz'] / p['lo_hz']), 0, 1)
    return ModeOut(pal(np.clip(sc + p['spread'] * (ctx.y - 0.5), 0, 1)))


PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@mode('key_colour', 'Key / Chord Colour', 'New',
      'Chroma analysis finds the dominant pitch centre over a sliding window; the whole scene takes that '
      "key's hue (circle of fifths so related keys look related). Dots keep a hint of their own note.",
      [Param('window_s', 'Window s', 'float', 2.0, 0.1, 10, log=True),
       Param('order', 'Circle', 'choice', 'fifths', choices=['fifths', 'chromatic']),
       Param('detail', 'Per-dot note detail', 'float', 0.25, 0, 1)],
      palette='Rainbow (HSV)')
def m_key(ctx, p, pal, up):
    if ctx.feat is None:
        return ModeOut(pal(np.zeros(ctx.n)))
    ch = ctx.feat_window('chroma', p['window_s'])
    pcs = np.arange(12)
    pos = (pcs * 7 % 12) / 12 if p['order'] == 'fifths' else pcs / 12
    z = (ch * np.exp(2j * np.pi * pos)).sum()
    key = (np.angle(z) / (2 * np.pi)) % 1.0
    midi = midi_of(ctx.freq)
    dpc = np.floor(midi) % 12
    dpos = (dpc * 7 % 12) / 12 if p['order'] == 'fifths' else dpc / 12
    d = ((dpos - key + 0.5) % 1.0) - 0.5
    return ModeOut(pal((key + p['detail'] * d) % 1.0))


CONSONANCE_RANK = np.array([0, 5, 4, 3, 2, 1, 6, 1, 2, 3, 4, 5]) / 6.0


@mode('consonance', 'Consonance', 'New',
      'Each dot is coloured by how consonant its pitch is against the current root (strongest note): unison/'
      'fifths at one end, seconds and the tritone at the other. Out-of-tune dots lean dissonant.',
      [Param('root', 'Root from', 'choice', 'chroma', choices=['chroma', 'strongest peak']),
       Param('window_s', 'Chroma window s', 'float', 0.5, 0.05, 5, log=True)],
      palette='Consonance')
def m_consonance(ctx, p, pal, up):
    if p['root'] == 'chroma' and ctx.feat is not None:
        root = int(np.argmax(ctx.feat_window('chroma', p['window_s'])))
    else:
        sel = (ctx.freq > 55) & (ctx.freq < 2000)
        root = int(np.floor(midi_of(ctx.freq[np.argmax(np.where(sel, ctx.amp, 0))]))) % 12
    midi = midi_of(ctx.freq)
    iv = (np.round(midi) - root) % 12
    cents = np.abs(midi - np.round(midi))
    s = np.clip(CONSONANCE_RANK[iv.astype(int)] + cents * 0.8, 0, 1)
    return ModeOut(pal(s))


@mode('time_cycle', 'Time Cycle', 'New', 'The palette scrolls through time (optionally locked to the beat), offset by position.',
      [Param('speed', 'Cycles per second', 'float', 0.2, -4, 4), Param('per_beat', 'Cycles per beat instead', 'bool', False),
       Param('spatial_y', 'Height offset', 'float', 0.5, -4, 4), Param('spatial_x', 'Pan offset', 'float', 0.0, -4, 4)],
      palette='Neon')
def m_time_cycle(ctx, p, pal, up):
    base = ctx.beat_pos() * p['speed'] if p['per_beat'] else ctx.t * p['speed']
    return ModeOut(pal((base + p['spatial_y'] * ctx.y + p['spatial_x'] * ctx.x * 0.5) % 1.0))


def _hash3(i, j, k):
    h = np.sin(i * 127.1 + j * 311.7 + k * 74.7) * 43758.5453
    return h - np.floor(h)


def value_noise3(x, y, z):
    xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
    xf, yf, zf = x - xi, y - yi, z - zi
    u, v, w = xf * xf * (3 - 2 * xf), yf * yf * (3 - 2 * yf), zf * zf * (3 - 2 * zf)
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wgt = (u if dx else 1 - u) * (v if dy else 1 - v) * (w if dz else 1 - w)
                out = out + wgt * _hash3(xi + dx, yi + dy, zi + dz)
    return out


def fbm(x, y, z, octaves=4):
    s, a, f, tot = 0, 0.5, 1.0, 0
    for _ in range(octaves):
        s = s + a * value_noise3(x * f, y * f, z * f)
        tot += a
        a *= 0.5
        f *= 2.03
    return s / tot


@mode('flow_field', 'Flow Field Marble', 'New',
      'Domain-warped fractal noise sampled at each dot and flowing through time: an evolving marble / '
      'nebula texture that the dots reveal.',
      [Param('scale', 'Scale', 'float', 2.5, 0.2, 12, log=True), Param('speed', 'Flow speed', 'float', 0.15, 0, 2),
       Param('warp', 'Warp', 'float', 0.8, 0, 3), Param('octaves', 'Octaves', 'int', 4, 1, 6),
       Param('audio_drive', 'Loudness drives flow', 'float', 0.5, 0, 2)],
      palette='Aurora')
def m_flow(ctx, p, pal, up):
    X, Y = screen_xy(ctx)
    z = ctx.t * p['speed'] + (ctx.feat_cumsum('rms') * p['audio_drive'] if ctx.feat is not None else 0)
    sx, sy = X * p['scale'], Y * p['scale']
    wx = fbm(sx + 1.7, sy + 9.2, z, 2)
    wy = fbm(sx + 8.3, sy + 2.8, z, 2)
    n = fbm(sx + p['warp'] * 4 * wx, sy + p['warp'] * 4 * wy, z * 0.7, p['octaves'])
    return ModeOut(pal(np.clip((n - 0.25) * 2, 0, 1) if not pal.cyclic else n * 2))


@mode('beat_pulse', 'Beat Pulse', 'New',
      'Tracked beats step the palette forward (one colour per beat) and every beat flashes the dots.',
      [Param('step', 'Palette step per beat', 'float', 0.17, 0, 1), Param('spread', 'Height spread', 'float', 0.2, 0, 1),
       Param('flash', 'Flash boost', 'float', 1.2, 0, 5), Param('decay', 'Flash decay (beats)', 'float', 0.25, 0.02, 1),
       Param('downbeat_every', 'Bar length (beats)', 'int', 4, 1, 16, tip='Every Nth beat gets a double flash.')],
      palette='Synthwave')
def m_beat_pulse(ctx, p, pal, up):
    bp = ctx.beat_pos()
    k = np.floor(bp)
    ph = bp - k
    s = (k * p['step'] + p['spread'] * ctx.y)
    fl = np.exp(-ph / max(p['decay'], 1e-3))
    if p['downbeat_every'] > 1 and int(k) % p['downbeat_every'] == 0:
        fl *= 2
    rgb = pal(s % 1.0)
    return ModeOut(adjust_lightness(rgb, 0.15 * fl), gain=np.full(ctx.n, 1 + p['flash'] * fl, np.float32))


@mode('attack_decay', 'Attack / Decay', 'New',
      'Compares every dot with the previous moment: rising energy (attacks) glows warm, sustaining energy '
      'neutral, decaying energy cool. Shows the motion of sound.',
      [Param('sensitivity', 'Sensitivity', 'float', 0.08, 0.01, 1, log=True), Param('lookback', 'Look-back frames', 'int', 2, 1, 16)],
      palette='Attack / Decay')
def m_attack(ctx, p, pal, up):
    prev = ctx.prev(p['lookback'])
    if prev is None:
        return ModeOut(pal(np.full(ctx.n, 0.5)))
    d = (np.log(ctx.amp + 1e-6) - np.log(prev.amp + 1e-6)) / max(p['lookback'], 1)
    return ModeOut(pal(np.clip(0.5 + d * p['sensitivity'], 0, 1)))


@mode('density', 'Cluster Density', 'New',
      'Dots are coloured by how crowded their neighbourhood is (amplitude-weighted 2D density): lone dots '
      'one colour, dense clusters another.',
      [Param('grid', 'Grid resolution', 'int', 48, 8, 160), Param('blur', 'Blur', 'float', 1.5, 0, 6), Param('gamma', 'Gamma', 'float', 0.7, 0.2, 3)],
      palette='Viridis')
def m_density(ctx, p, pal, up):
    from scipy.ndimage import gaussian_filter
    g = int(p['grid'])
    gx = np.clip(((ctx.x + 1) / 2 * g).astype(int), 0, g - 1)
    gy = np.clip((ctx.y * g).astype(int), 0, g - 1)
    H = np.zeros((g, g))
    np.add.at(H, (gy, gx), ctx.level * ctx.visible)
    if p['blur'] > 0:
        H = gaussian_filter(H, p['blur'])
    H /= H.max() + 1e-9
    return ModeOut(pal(H[gy, gx] ** p['gamma']))


@mode('stems', 'Instrument Stems', 'New',
      'Demucs separates vocals / drums / bass / other on the GPU; every dot is coloured by which instrument '
      'owns that frequency at that moment (blended in OKLab when they share it).',
      [Param('vocals', 'Vocals colour', 'color', [1.0, 0.3, 0.45]), Param('drums', 'Drums colour', 'color', [1.0, 0.85, 0.3]),
       Param('bass', 'Bass colour', 'color', [0.3, 0.55, 1.0]), Param('other', 'Other colour', 'color', [0.35, 1.0, 0.7]),
       Param('sharpness', 'Sharpness', 'float', 2.0, 0.5, 8), Param('solo', 'Solo', 'choice', 'none', choices=['none', 'vocals', 'drums', 'bass', 'other'])],
      native=True, needs=('stems',))
def m_stems(ctx, p, pal, up):
    r = ctx.stem_ratios()
    if r is None:
        return ModeOut(np.full((ctx.n, 3), 0.6, np.float32))
    w = np.maximum(r, 1e-4) ** p['sharpness']
    cols = np.array([p['vocals'], p['drums'], p['bass'], p['other']], np.float32)
    rgb = weighted_oklab(cols, w)
    s = (w / w.sum(1, keepdims=True)) @ np.array([0.0, 1 / 3, 2 / 3, 1.0])
    gain = None
    if p['solo'] != 'none':
        k = ['vocals', 'drums', 'bass', 'other'].index(p['solo'])
        gain = np.clip(r[:, k] * 1.5, 0, 1).astype(np.float32)
    return ModeOut(pick(rgb, s, pal, up), gain=gain)


@mode('vowel', 'Vowel / Mouth Shape', 'New',
      "Tracks the singer's vowel (A E I O U and blends between) from the isolated vocal and colours the dots by "
      'mouth shape. Choose a colour per vowel; consonants/rests hold the last vowel or fall back to neutral.',
      [Param('A', 'A (ah)', 'color', DEFAULT_VOWEL_COLORS['A']), Param('E', 'E (eh)', 'color', DEFAULT_VOWEL_COLORS['E']),
       Param('I', 'I (ee)', 'color', DEFAULT_VOWEL_COLORS['I']), Param('O', 'O (oh)', 'color', DEFAULT_VOWEL_COLORS['O']),
       Param('U', 'U (oo)', 'color', DEFAULT_VOWEL_COLORS['U']),
       Param('apply_to', 'Apply to', 'choice', 'all dots', choices=['all dots', 'vocal dots']),
       Param('unvoiced', 'Between vowels', 'choice', 'hold', choices=['hold', 'neutral', 'fade']),
       Param('neutral', 'Neutral colour', 'color', [0.8, 0.8, 0.85]),
       Param('openness_brightness', 'Mouth openness -> brightness', 'float', 0.0, 0, 1),
       Param('sharpness', 'Vowel sharpness', 'float', 1.0, 0.3, 4, tip='>1 snaps to the nearest vowel; <1 blends more.')],
      native=True, needs=('vowel',))
def m_vowel(ctx, p, pal, up):
    v = ctx.vowel_state(p['unvoiced'] == 'hold')
    if v is None:
        return ModeOut(np.tile(np.array(p['neutral'], np.float32), (ctx.n, 1)))
    w, voiced, openness = v
    w = w ** p['sharpness']
    w = w / w.sum()
    cols = np.array([p[k] for k in VOWELS], np.float32)
    col = weighted_oklab(cols, w[None, :])[0]
    if p['unvoiced'] in ('neutral', 'fade'):
        col = mix_oklab(np.array(p['neutral'], np.float32), col, float(np.clip(voiced, 0, 1)))
    s = float(w @ np.linspace(0, 1, 5))
    native = np.tile(col, (ctx.n, 1)).astype(np.float32)
    rgb = pick(native, np.full(ctx.n, s), pal, up)
    gain = None
    if p['apply_to'] == 'vocal dots':
        r = ctx.stem_ratios()
        voc = (r[:, 0] > 0.45) if r is not None else ((ctx.freq > 150) & (ctx.freq < 5000) & (np.abs(ctx.pan) < 0.25))
        neutral = np.broadcast_to(np.array(p['neutral'], np.float32), rgb.shape)
        rgb = np.where(voc[:, None], rgb, neutral)
    if p['openness_brightness'] > 0:
        gain = np.full(ctx.n, 1 + p['openness_brightness'] * (2 * openness - 1) * 0.8, np.float32)
    if p['unvoiced'] == 'fade':
        gain = (gain if gain is not None else np.ones(ctx.n, np.float32)) * (0.35 + 0.65 * voiced)
    return ModeOut(rgb, gain=gain)


@mode('note_octave', 'Note + Octave', 'New',
      'Hue from the note name, lightness from the octave: bass notes deep and dark, treble notes pale.',
      [Param('step', 'Lightness per octave', 'float', 0.07, 0, 0.2), Param('center_octave', 'Middle octave', 'float', 4, 1, 8)],
      native=True)
def m_note_octave(ctx, p, pal, up):
    midi = midi_of(ctx.freq)
    octv = midi / 12 - 1
    base = pick(note_hue(midi), (midi % 12) / 12, pal, up)
    return ModeOut(adjust_lightness(base, (octv - p['center_octave']) * p['step']))


@mode('pitch_spiral', 'Pitch Spiral', 'New',
      'Continuous log-frequency through a cyclic palette: every octave goes once round the palette, so octaves '
      'of the same note share a colour while the whole spectrum shows as a smooth helix.',
      [Param('turns_per_octave', 'Turns per octave', 'float', 1.0, 0.25, 4), Param('phase', 'Phase', 'float', 0.0, 0, 1)],
      palette='Vaporwave')
def m_pitch_spiral(ctx, p, pal, up):
    return ModeOut(pal(np.log2(np.maximum(ctx.freq, 1) / 16.35) * p['turns_per_octave'] + p['phase']))


@mode('solid', 'Solid Colour', 'New', 'A single colour (useful as a tween target).',
      [Param('color', 'Colour', 'color', [1.0, 0.55, 0.2])], native=True)
def m_solid(ctx, p, pal, up):
    return ModeOut(pick(np.tile(np.array(p['color'], np.float32), (ctx.n, 1)), np.full(ctx.n, 0.5), pal, up))


CATEGORIES = ['Ported', 'New']


def interpolate_params(mode_obj, a, b, u):
    """Lerp numeric params, OKLab-mix colours, switch discrete params at u=0.5."""
    out = {}
    for prm in mode_obj.params:
        va, vb = a.get(prm.key, prm.default), b.get(prm.key, prm.default)
        if prm.kind in ('float', 'int'):
            v = va + (vb - va) * u
            out[prm.key] = int(round(v)) if prm.kind == 'int' else float(v)
        elif prm.kind == 'color':
            out[prm.key] = [float(c) for c in oklab_to_srgb(srgb_to_oklab(np.array(va, np.float32)) * (1 - u)
                                                            + srgb_to_oklab(np.array(vb, np.float32)) * u)]
        elif prm.kind == 'zones' and va and vb and len(va) == len(vb):
            out[prm.key] = [{k: (za[k] + (zb[k] - za[k]) * u if k in ('x', 'y', 'r') else
                                 (za[k] if u < 0.5 else zb[k])) for k in za} for za, zb in zip(va, vb)]
        else:
            out[prm.key] = va if u < 0.5 else vb
    return out
