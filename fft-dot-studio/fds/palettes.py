"""Gradient palettes: editable stops interpolated in OKLab, with presets and LUT caching."""
import json
import os
import random
from functools import lru_cache

import numpy as np

from .colorspace import hex_to_rgb, hsv_to_rgb, oklab_to_srgb, oklch_to_srgb, srgb_to_oklab

LUT_SIZE = 512


def _hex_stops(hexes, cyclic=False):
    n = len(hexes)
    denom = n if cyclic else max(n - 1, 1)
    return [(i / denom, hex_to_rgb(h)) for i, h in enumerate(hexes)]


PRESETS = {
    'Rainbow (HSV)': ([(i / 12, list(hsv_to_rgb(i / 12))) for i in range(12)], True),
    'Inferno': (_hex_stops(['#000004', '#420a68', '#932667', '#dd513a', '#fca50a', '#fcffa4']), False),
    'Magma': (_hex_stops(['#000004', '#3b0f70', '#8c2981', '#de4968', '#fe9f6d', '#fcfdbf']), False),
    'Viridis': (_hex_stops(['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']), False),
    'Plasma': (_hex_stops(['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921']), False),
    'Turbo': (_hex_stops(['#30123b', '#4686fb', '#1ae4b6', '#a2fc3c', '#faba39', '#e4460a', '#7a0403']), False),
    'Fire': (_hex_stops(['#200000', '#7a0000', '#d42000', '#ff7800', '#ffd200', '#ffffff']), False),
    'Ice': (_hex_stops(['#0a1440', '#0a3a8e', '#1e78c8', '#64c8ff', '#e6faff']), False),
    'Ocean': (_hex_stops(['#02253a', '#004f73', '#00889a', '#3cc0aa', '#c8f0dc']), False),
    'Sunset': (_hex_stops(['#3a0b5c', '#8b1d6c', '#d2365a', '#f2724b', '#fcb860', '#fff1a8']), False),
    'Synthwave': (_hex_stops(['#3b1f8c', '#ab1f65', '#ff4f69', '#ff8142', '#ffda45']), False),
    'Neon': (_hex_stops(['#ff2bd6', '#7a2bff', '#2bd9ff', '#2bff88'], cyclic=True), True),
    'Vaporwave': (_hex_stops(['#ff71ce', '#01cdfe', '#05ffa1', '#b967ff', '#fffb96'], cyclic=True), True),
    'Cyberpunk': (_hex_stops(['#00f0ff', '#ff003c', '#fcee0a'], cyclic=True), True),
    'Aurora': (_hex_stops(['#00ff9f', '#00b8ff', '#2050ff', '#bd00ff'], cyclic=True), True),
    'Candy': (_hex_stops(['#ff6fa8', '#ffd36e', '#7af0c9', '#8fa8ff'], cyclic=True), True),
    'Pastel': (_hex_stops(['#ffb3ba', '#ffdfba', '#ffffba', '#baffc9', '#bae1ff'], cyclic=True), True),
    'Gold': (_hex_stops(['#3a2200', '#8a5a00', '#d4a017', '#ffe58a', '#fffbe6']), False),
    'Forest': (_hex_stops(['#16391f', '#2f6b3a', '#3f8f4a', '#9bd36a', '#e8f5a0']), False),
    'Greyscale': (_hex_stops(['#202020', '#ffffff']), False),
    'White': (_hex_stops(['#ffffff', '#ffffff']), False),
    'Stereo (L blue / R red)': (_hex_stops(['#2f7dff', '#ffffff', '#ff3b3b']), False),
    'Consonance': (_hex_stops(['#3dff7a', '#c8ff3d', '#ffd23d', '#ff7a3d', '#ff2d55']), False),
    'Radial Glow': (_hex_stops(['#fff4c2', '#ff9a3d', '#ff2d55', '#7a1fd6', '#2a0c8a']), False),
    'Attack / Decay': (_hex_stops(['#2f7dff', '#ffffff', '#ff8c1a']), False),
    'Heat': (_hex_stops(['#1b1b6e', '#7a1fd6', '#ff2d55', '#ffb000', '#ffffff']), False),
}


def _key(stops, cyclic):
    return (tuple((round(float(p), 5), tuple(round(float(x), 5) for x in c)) for p, c in stops), bool(cyclic))


@lru_cache(maxsize=256)
def _lut_cached(key):
    stops, cyclic = key
    pos = np.array([p for p, _ in stops], dtype=np.float64)
    labs = srgb_to_oklab(np.array([c for _, c in stops], dtype=np.float32)).astype(np.float64)
    order = np.argsort(pos, kind='stable')
    pos, labs = pos[order], labs[order]
    xs = np.linspace(0.0, 1.0, LUT_SIZE, endpoint=not cyclic)
    if len(pos) == 1:
        out = np.repeat(labs[:1], LUT_SIZE, axis=0)
    elif cyclic:
        pos_ext = np.concatenate([pos - 1.0, pos, pos + 1.0])
        lab_ext = np.concatenate([labs, labs, labs])
        out = np.stack([np.interp(xs, pos_ext, lab_ext[:, k]) for k in range(3)], axis=1)
    else:
        out = np.stack([np.interp(xs, pos, labs[:, k]) for k in range(3)], axis=1)
    rgb = oklab_to_srgb(out.astype(np.float32))
    rgb.setflags(write=False)
    return rgb


class Palette:
    def __init__(self, stops=None, cyclic=False, name='Custom'):
        if stops is None:
            stops, cyclic = PRESETS['Rainbow (HSV)']
            name = 'Rainbow (HSV)'
        self.stops = [(float(p), [float(x) for x in c]) for p, c in stops]
        self.cyclic = bool(cyclic)
        self.name = name

    @classmethod
    def preset(cls, name):
        stops, cyclic = PRESETS[name]
        return cls([(p, list(c)) for p, c in stops], cyclic, name)

    def copy(self):
        return Palette([(p, list(c)) for p, c in self.stops], self.cyclic, self.name)

    def lut(self):
        return _lut_cached(_key(self.stops, self.cyclic))

    def sample(self, s):
        return sample_lut(self.lut(), self.cyclic, s)

    def mean_color(self):
        return self.lut().mean(axis=0)

    def reversed(self):
        return Palette([(1.0 - p if not self.cyclic else (1.0 - p) % 1.0, list(c)) for p, c in self.stops],
                       self.cyclic, self.name + ' (rev)')

    def to_dict(self):
        return {'name': self.name, 'cyclic': self.cyclic, 'stops': [[p, list(c)] for p, c in self.stops]}

    @classmethod
    def from_dict(cls, d):
        return cls([(p, c) for p, c in d['stops']], d.get('cyclic', False), d.get('name', 'Custom'))

    def __eq__(self, other):
        return isinstance(other, Palette) and _key(self.stops, self.cyclic) == _key(other.stops, other.cyclic)


def sample_lut(lut, cyclic, s):
    s = np.asarray(s, dtype=np.float32)
    n = lut.shape[0]
    if cyclic:
        idx = np.floor((s % 1.0) * n).astype(np.int64) % n
    else:
        idx = np.clip(np.round(np.nan_to_num(s) * (n - 1)), 0, n - 1).astype(np.int64)
    return lut[idx]


def mix_luts(lut_a, lut_b, t):
    if t <= 0:
        return lut_a
    if t >= 1:
        return lut_b
    la, lb = srgb_to_oklab(lut_a), srgb_to_oklab(lut_b)
    return oklab_to_srgb(la + (lb - la) * float(t))


def random_palette(seed=None, n=None, cyclic=None):
    rng = random.Random(seed)
    n = n or rng.randint(3, 5)
    cyclic = rng.random() < 0.4 if cyclic is None else cyclic
    h0 = rng.random()
    scheme = rng.choice(['analogous', 'triad', 'complement', 'wide'])
    spread = {'analogous': 0.18, 'triad': 0.66, 'complement': 0.5, 'wide': 1.0}[scheme]
    stops = []
    for i in range(n):
        t = i / (n if cyclic else max(n - 1, 1))
        L = 0.55 + 0.35 * (t if not cyclic else 0.5 + 0.4 * np.sin(2 * np.pi * t))
        C = 0.12 + 0.1 * rng.random()
        h = h0 + spread * t + rng.uniform(-0.03, 0.03)
        stops.append((t, [float(x) for x in oklch_to_srgb(L, C, h)]))
    return Palette(stops, cyclic, f'Random {scheme}')


def user_preset_path():
    base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    return os.path.join(base, 'fft-dot-studio', 'palettes.json')


def load_user_presets():
    try:
        with open(user_preset_path(), 'r', encoding='utf-8') as f:
            return {k: Palette.from_dict(v) for k, v in json.load(f).items()}
    except (OSError, ValueError, KeyError):
        return {}


def save_user_preset(pal: Palette, name: str):
    presets = {k: v.to_dict() for k, v in load_user_presets().items()}
    p = pal.copy()
    p.name = name
    presets[name] = p.to_dict()
    path = user_preset_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(presets, f, indent=1)
