"""Project model: tracks laid end to end, keyframes with easing, render/analysis settings, JSON I/O."""
import copy
import json
import os
import uuid

import numpy as np

from .analysis import ANALYSIS_DEFAULTS, sanitize_settings
from .modes import MODES, MODE_ORDER
from .palettes import Palette

PROJECT_VERSION = 1

RENDER_DEFAULTS = {
    # frequency axis
    'freq_min': 20.0, 'freq_max': 40000.0, 'freq_scale': 'log',
    # amplitude
    'amp_scale': 'linear', 'db_floor': -60.0, 'normalise': 'frame', 'running_release_s': 2.0,
    'threshold': 0.01,
    # dot selection
    'peak_pick': False, 'peak_min_dist': 0.01, 'max_dots': 0,
    # temporal
    'smooth_attack_ms': 0.0, 'smooth_release_ms': 0.0, 'trail_ms': 25.0,
    # appearance
    'dot_size': 5.0, 'dot_shape': 'square', 'brightness': 8.0, 'gamma': 1.0,
    'glow': 0.0, 'glow_radius': 1.0, 'tonemap': 'clip', 'saturation': 1.0, 'background': [0.0, 0.0, 0.0],
    # geometry
    'pan_mode': 'level', 'pan_spread': 1.0, 'flip_x': False, 'flip_y': False, 'mirror_x': False, 'mirror_y': False,
    'edge_margin': 0.02,
}
RENDER_CHOICES = {
    'freq_scale': ['log', 'mel', 'linear'], 'amp_scale': ['linear', 'db'],
    'normalise': ['frame', 'running', 'global'], 'dot_shape': ['square', 'round', 'glow'],
    'tonemap': ['clip', 'soft'], 'pan_mode': ['level', 'phase'],
}
ANIMATABLE = ['brightness', 'gamma', 'dot_size', 'trail_ms', 'glow', 'glow_radius', 'saturation', 'threshold',
              'db_floor', 'background', 'pan_spread']
RENDER_PRESETS = {
    'Classic v5': dict(RENDER_DEFAULTS),
    'Punchy beats': dict(RENDER_DEFAULTS, amp_scale='db', db_floor=-48.0, normalise='running', running_release_s=1.5,
                         smooth_release_ms=60.0, trail_ms=45.0, dot_size=6.0, dot_shape='round', brightness=1.6,
                         gamma=1.6, glow=0.45, tonemap='soft', threshold=0.003),
    'Smooth bass': dict(RENDER_DEFAULTS, amp_scale='db', db_floor=-60.0, normalise='global', smooth_attack_ms=15.0,
                        smooth_release_ms=220.0, trail_ms=120.0, dot_size=7.0, dot_shape='glow', brightness=0.9,
                        gamma=1.3, glow=0.6, tonemap='soft', threshold=0.002),
    'Fine sparkle': dict(RENDER_DEFAULTS, amp_scale='db', db_floor=-54.0, normalise='running', peak_pick=True,
                         peak_min_dist=0.006, max_dots=900, trail_ms=80.0, dot_size=4.0, dot_shape='round',
                         brightness=1.4, gamma=1.2, glow=0.35, tonemap='soft', threshold=0.002),
}
RENDER_TIPS = {
    'freq_min': 'Lowest frequency at the bottom of the screen (Hz).',
    'freq_max': 'Frequency at the top of the screen. 40 kHz (original) leaves headroom above the 22 kHz audio limit.',
    'freq_scale': 'Vertical frequency axis: log (musical, original), mel (perceptual), linear.',
    'amp_scale': 'Dot brightness from linear amplitude (original) or decibels (reveals quiet detail).',
    'db_floor': 'In dB mode, amplitudes this far below the reference are black.',
    'normalise': 'Brightness reference: per frame (original: every frame as bright as the last), running '
                 'loudness (beats stand out, quiet parts dim) or global track loudness.',
    'running_release_s': 'How quickly the running loudness reference falls after a loud passage.',
    'threshold': 'Dots quieter than this fraction of the reference are hidden.',
    'peak_pick': 'Keep only spectral peaks (local maxima) instead of every FFT bin.',
    'peak_min_dist': 'Minimum distance between picked peaks in normalised screen space.',
    'max_dots': 'Keep only the N loudest dots per frame (0 = unlimited).',
    'smooth_attack_ms': 'Temporal smoothing when a bin gets louder (0 = instant).',
    'smooth_release_ms': 'Temporal smoothing when a bin gets quieter (longer = dots linger).',
    'trail_ms': 'Half-life of the fading trail (original fade ~25 ms).',
    'dot_size': 'Dot size in pixels at 1080p (scales with output height).',
    'dot_shape': 'Square points (original), soft round, or round with a glow halo.',
    'brightness': 'Overall dot brightness multiplier.',
    'gamma': 'Amplitude curve: >1 makes quiet dots darker and loud dots pop.',
    'glow': 'Bloom strength (post-process).', 'glow_radius': 'Bloom size.',
    'tonemap': 'Clip (original hard saturation) or soft filmic roll-off.',
    'saturation': 'Colour saturation of the final image.', 'background': 'Background colour.',
    'pan_mode': 'Horizontal position from L/R level difference (original) or inter-channel phase difference '
                '(spreads wide/decorrelated sounds, centres mono).',
    'pan_spread': 'Horizontal spread multiplier.',
    'flip_x': 'Swap left and right.', 'flip_y': 'Put treble at the bottom.',
    'mirror_x': 'Kaleidoscope: also draw every dot mirrored left/right.',
    'mirror_y': 'Also draw every dot mirrored top/bottom (bass in the middle).',
    'edge_margin': 'Keeps dots fully on screen (fraction of the half-width).',
}

ALIGN_MODES = ['centred', 'live', 'custom']
ALIGN_LABELS = ['Centred (default)', 'Live-style (window ends at t, like v5)', 'Custom offset']


def align_shifts(align, band_windows, hop, sr, feat_n):
    """Per-band frame shifts and the feature-time shift (s) for an analysis alignment setting.
    A frame displayed at time t analyses audio centred on t + shift."""
    mode = align.get('mode', 'centred')
    if mode == 'live':
        offs = [-(n / 2) / sr for n in band_windows]
        feat = -(feat_n / 2) / sr
    elif mode == 'custom':
        o = float(align.get('offset_ms', 0.0)) / 1000.0
        offs = [o for _ in band_windows]
        feat = o
    else:
        offs = [0.0 for _ in band_windows]
        feat = 0.0
    return tuple(round(o * sr / hop, 6) for o in offs), feat


EASINGS = ['linear', 'smooth', 'ease in', 'ease out', 'ease in-out', 'exponential', 'hold']


def ease(u, kind):
    u = float(min(max(u, 0.0), 1.0))
    if kind == 'smooth':
        return u * u * (3 - 2 * u)
    if kind == 'ease in':
        return u * u * u
    if kind == 'ease out':
        return 1 - (1 - u) ** 3
    if kind == 'ease in-out':
        return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2
    if kind == 'exponential':
        return (2 ** (10 * u) - 1) / 1023.0
    if kind == 'hold':
        return 0.0
    return u


class Keyframe:
    def __init__(self, time=0.0, mode='note_rainbow', easing='linear', transition='morph'):
        self.id = uuid.uuid4().hex[:8]
        self.time = float(time)
        self.easing = easing
        self.transition = transition
        self.mode = mode
        self.params_by_mode = {}
        self.palette = Palette.preset(MODES[mode].palette)
        self.use_palette = False
        self.render_overrides = {}
        self.label = ''

    @property
    def params(self):
        return MODES[self.mode].fill(self.params_by_mode.get(self.mode))

    def set_param(self, key, value):
        p = self.params
        p[key] = value
        self.params_by_mode[self.mode] = p

    def set_mode(self, mode, adopt_palette=True):
        if mode == self.mode:
            return
        self.mode = mode
        if adopt_palette:
            self.palette = Palette.preset(MODES[mode].palette)

    def copy(self):
        k = copy.deepcopy(self)
        k.id = uuid.uuid4().hex[:8]
        return k

    def to_dict(self):
        return {'id': self.id, 'time': self.time, 'easing': self.easing, 'transition': self.transition,
                'mode': self.mode, 'params_by_mode': copy.deepcopy(self.params_by_mode),
                'palette': self.palette.to_dict(), 'use_palette': self.use_palette,
                'render_overrides': copy.deepcopy(self.render_overrides), 'label': self.label}

    @classmethod
    def from_dict(cls, d):
        k = cls(d.get('time', 0.0), d.get('mode', 'note_rainbow') if d.get('mode') in MODES else 'note_rainbow',
                d.get('easing', 'linear'), d.get('transition', 'morph'))
        k.id = d.get('id', k.id)
        k.params_by_mode = {m: v for m, v in d.get('params_by_mode', {}).items() if m in MODES}
        k.palette = Palette.from_dict(d['palette']) if 'palette' in d else k.palette
        k.use_palette = bool(d.get('use_palette', False))
        k.render_overrides = {a: v for a, v in d.get('render_overrides', {}).items() if a in ANIMATABLE}
        k.label = d.get('label', '')
        return k


class Project:
    def __init__(self):
        self.path = None
        self.tracks = []                      # list of absolute file paths, in timeline order
        self.keyframes = [Keyframe(0.0, 'note_rainbow')]
        self.render = dict(RENDER_DEFAULTS)
        self.analysis = dict(ANALYSIS_DEFAULTS)
        self.align = {'mode': 'centred', 'offset_ms': 0.0}
        self.preview_latency = {'mode': 'measured', 'ms': 0.0}
        self.preview_fps = 30
        self.gap_s = 0.0
        self.use_demucs = True
        self.vocal_source = 'auto'
        self.export = {'fps': 30, 'width': 1920, 'height': 1080, 'codec': 'auto', 'audio': 'copy',
                       'quality': 18, 'path': ''}
        self.revision = 0

    # ---- keyframes
    def sorted_keys(self):
        self.keyframes.sort(key=lambda k: k.time)
        return self.keyframes

    def key_by_id(self, kid):
        for k in self.keyframes:
            if k.id == kid:
                return k
        return None

    def bracket(self, t):
        ks = self.sorted_keys()
        if t <= ks[0].time or len(ks) == 1:
            return ks[0], None, 0.0
        for i in range(len(ks) - 1):
            if ks[i].time <= t < ks[i + 1].time:
                k0, k1 = ks[i], ks[i + 1]
                span = max(k1.time - k0.time, 1e-9)
                return k0, k1, ease((t - k0.time) / span, k0.easing)
        return ks[-1], None, 0.0

    def render_at(self, t):
        k0, k1, u = self.bracket(t)
        r = dict(self.render)
        r.update(k0.render_overrides)
        if k1 is not None and u > 0:
            r1 = dict(self.render)
            r1.update(k1.render_overrides)
            for key in ANIMATABLE:
                a, b = r[key], r1[key]
                if isinstance(a, list):
                    r[key] = [float(x + (y - x) * u) for x, y in zip(a, b)]
                else:
                    r[key] = float(a + (b - a) * u)
        return r

    # ---- timeline
    def track_offsets(self, durations):
        offs, t = [], 0.0
        for d in durations:
            offs.append(t)
            t += d + self.gap_s
        return offs, max(0.0, t - self.gap_s) if durations else 0.0

    # ---- io
    def to_dict(self):
        return {'version': PROJECT_VERSION, 'tracks': [os.path.normpath(t) for t in self.tracks],
                'keyframes': [k.to_dict() for k in self.sorted_keys()], 'render': copy.deepcopy(self.render),
                'analysis': dict(self.analysis), 'align': dict(self.align),
                'preview_latency': dict(self.preview_latency), 'preview_fps': self.preview_fps,
                'gap_s': self.gap_s, 'use_demucs': self.use_demucs, 'vocal_source': self.vocal_source,
                'export': dict(self.export)}

    @classmethod
    def from_dict(cls, d, base_dir=None):
        p = cls()
        tracks = []
        for t in d.get('tracks', []):
            if base_dir and not os.path.isabs(t):
                t = os.path.normpath(os.path.join(base_dir, t))
            tracks.append(os.path.normpath(t))
        p.tracks = tracks
        p.keyframes = [Keyframe.from_dict(k) for k in d.get('keyframes', [])] or [Keyframe(0.0)]
        p.render = dict(RENDER_DEFAULTS)
        p.render.update({k: v for k, v in d.get('render', {}).items() if k in RENDER_DEFAULTS})
        p.analysis = sanitize_settings(d.get('analysis', {}))
        p.align.update(d.get('align', {}))
        p.preview_latency.update(d.get('preview_latency', {}))
        p.preview_fps = d.get('preview_fps', 30)
        p.gap_s = float(d.get('gap_s', 0.0))
        p.use_demucs = bool(d.get('use_demucs', True))
        p.vocal_source = d.get('vocal_source', 'auto')
        p.export.update({k: v for k, v in d.get('export', {}).items() if k in p.export})
        return p

    def save(self, path):
        data = self.to_dict()
        base = os.path.dirname(os.path.abspath(path))
        rel = []
        for t in data['tracks']:
            try:
                r = os.path.relpath(t, base)
                rel.append(t if r.startswith('..' + os.sep + '..' + os.sep + '..') else r)
            except ValueError:
                rel.append(t)
        data['tracks'] = rel
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
        self.path = path

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        p = cls.from_dict(d, os.path.dirname(os.path.abspath(path)))
        p.path = path
        return p


def default_keyframe_mode_list():
    return list(MODE_ORDER)


def freq_to_y(freq, fmin, fmax, scale):
    f = np.asarray(freq, dtype=np.float32)
    if scale == 'linear':
        return (f - fmin) / max(fmax - fmin, 1e-6)
    if scale == 'mel':
        m = lambda x: 2595.0 * np.log10(1 + np.asarray(x) / 700.0)
        return (m(f) - m(fmin)) / max(m(fmax) - m(fmin), 1e-6)
    return np.log(np.maximum(f, 1e-3) / fmin) / np.log(fmax / fmin)


def y_to_freq(y, fmin, fmax, scale):
    y = np.asarray(y, dtype=np.float64)
    if scale == 'linear':
        return fmin + y * (fmax - fmin)
    if scale == 'mel':
        m = lambda x: 2595.0 * np.log10(1 + x / 700.0)
        mel = m(fmin) + y * (m(fmax) - m(fmin))
        return 700.0 * (10 ** (mel / 2595.0) - 1)
    return fmin * (fmax / fmin) ** y
