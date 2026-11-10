"""Vectorised colour maths: sRGB <-> linear <-> OKLab, HSV, OKLCh."""
import numpy as np

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], dtype=np.float64)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], dtype=np.float64)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((np.maximum(c, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float32), 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


def linear_to_oklab(rgb):
    lms = np.asarray(rgb, dtype=np.float64) @ _M1.T
    return (np.cbrt(lms) @ _M2.T).astype(np.float32)


def oklab_to_linear(lab):
    lms_ = np.asarray(lab, dtype=np.float64) @ _M2_INV.T
    return ((lms_ ** 3) @ _M1_INV.T).astype(np.float32)


def srgb_to_oklab(c):
    return linear_to_oklab(srgb_to_linear(c))


def oklab_to_srgb(lab):
    return np.clip(linear_to_srgb(oklab_to_linear(lab)), 0.0, 1.0)


def mix_oklab(a, b, t):
    """Mix sRGB colours a and b (broadcastable (...,3)) in OKLab by t (scalar or (...,))."""
    t = np.asarray(t, dtype=np.float32)
    if t.ndim:
        t = t[..., None]
    la, lb = srgb_to_oklab(a), srgb_to_oklab(b)
    return oklab_to_srgb(la + (lb - la) * t)


def weighted_oklab(colors, weights):
    """colors (K,3) sRGB, weights (N,K) -> (N,3) sRGB mixed in OKLab."""
    lab = srgb_to_oklab(np.asarray(colors, dtype=np.float32))
    w = np.asarray(weights, dtype=np.float32)
    w = w / np.maximum(w.sum(axis=-1, keepdims=True), 1e-9)
    return oklab_to_srgb(w @ lab)


def hsv_to_rgb(h, s=1.0, v=1.0):
    """Vectorised colorsys.hsv_to_rgb. h in turns (any real), returns (...,3)."""
    h = np.asarray(h, dtype=np.float32) % 1.0
    s = np.broadcast_to(np.asarray(s, dtype=np.float32), h.shape)
    v = np.broadcast_to(np.asarray(v, dtype=np.float32), h.shape)
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i.astype(np.int32) % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def oklch_to_srgb(L, C, h_turns):
    L = np.asarray(L, dtype=np.float32)
    C = np.asarray(C, dtype=np.float32)
    h = np.asarray(h_turns, dtype=np.float32) * 2 * np.pi
    lab = np.stack(np.broadcast_arrays(L, C * np.cos(h), C * np.sin(h)), axis=-1)
    return oklab_to_srgb(lab)


def desaturate(rgb, amount):
    """Pull sRGB colours towards their OKLab lightness grey by amount (scalar or (N,))."""
    lab = srgb_to_oklab(rgb)
    a = np.asarray(amount, dtype=np.float32)
    if a.ndim:
        a = a[..., None]
    lab[..., 1:] *= (1.0 - np.clip(a, 0, 1))
    return oklab_to_srgb(lab)


def adjust_lightness(rgb, delta):
    lab = srgb_to_oklab(rgb)
    d = np.asarray(delta, dtype=np.float32)
    lab[..., 0] = np.clip(lab[..., 0] + d, 0, 1)
    return oklab_to_srgb(lab)


def hex_to_rgb(h):
    h = h.lstrip('#')
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def rgb_to_hex(c):
    return '#' + ''.join(f'{int(round(np.clip(x, 0, 1) * 255)):02x}' for x in c[:3])
