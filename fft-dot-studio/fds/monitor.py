"""Pick the smaller monitor (the user's 2880x1620 side screen) via Win32 EnumDisplayMonitors."""
import ctypes
import sys


def set_dpi_aware():
    if sys.platform != 'win32':
        return
    try:
        fn = ctypes.windll.user32.SetProcessDpiAwarenessContext
        fn.restype = ctypes.c_bool
        fn.argtypes = [ctypes.c_void_p]
        fn(ctypes.c_void_p(-4))   # per-monitor aware v2, same as GLFW uses
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass


def monitors():
    """List of dicts: monitor rect, work rect, primary flag (physical pixels)."""
    if sys.platform != 'win32':
        return []
    from ctypes import wintypes

    class MONITORINFO(ctypes.Structure):
        _fields_ = [('cbSize', wintypes.DWORD), ('rcMonitor', wintypes.RECT),
                    ('rcWork', wintypes.RECT), ('dwFlags', wintypes.DWORD)]
    out = []
    proc_t = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                                wintypes.LPARAM)

    def cb(hmon, _hdc, _rect, _lp):
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        ctypes.windll.user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
        r, w = mi.rcMonitor, mi.rcWork
        out.append({'rect': (r.left, r.top, r.right - r.left, r.bottom - r.top),
                    'work': (w.left, w.top, w.right - w.left, w.bottom - w.top),
                    'primary': bool(mi.dwFlags & 1)})
        return 1
    ctypes.windll.user32.EnumDisplayMonitors(None, None, proc_t(cb), 0)
    return out


def smallest_monitor():
    mons = monitors()
    if not mons:
        return None
    return min(mons, key=lambda m: m['rect'][2] * m['rect'][3])
