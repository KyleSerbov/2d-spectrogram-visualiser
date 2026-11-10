"""Timeline panel: ruler, keyframe lane, colour lane, track lane with waveforms, beat + vowel lanes."""
import numpy as np
from imgui_bundle import imgui

from ..analysis import FEAT_FPS
from ..project import EASINGS
from ..vowel import DEFAULT_VOWEL_COLORS, VOWELS
from .widgets import V2, col32, keyframe_swatch

RULER_H, KF_H, COLOR_H, TRACK_H, BEAT_H, VOWEL_H = 22.0, 26.0, 14.0, 64.0, 10.0, 12.0
STEPS = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
VOWEL_COLS = np.array([DEFAULT_VOWEL_COLORS[v] for v in VOWELS], np.float32)


def fmt_time(t, frac=True):
    t = max(0.0, t)
    m = int(t // 60)
    s = t - 60 * m
    return f'{m:d}:{s:05.2f}' if frac else f'{m:d}:{int(s):02d}'


class Timeline:
    def __init__(self, app):
        self.app = app
        self.t0 = 0.0
        self.pps = None
        self.fit_next = True
        self.auto_fit = True
        self.drag_kf = None
        self.drag_orig = 0.0
        self.drag_mx = 0.0
        self.scrubbing = False
        self.ctx_kf = None
        self.snap = True
        self.rect = None

    def x_of(self, t, x0):
        return x0 + (t - self.t0) * self.pps

    def t_of(self, x, x0):
        return self.t0 + (x - x0) / self.pps

    def draw(self):
        app = self.app
        eng = app.engine
        total = eng.total_duration()
        span = max(total, 1.0)
        # toolbar
        if imgui.button('Fit'):
            self.auto_fit = True
        imgui.same_line()
        if imgui.button('+ Keyframe (K)'):
            app.add_keyframe(app.playhead)
        imgui.same_line()
        _c, self.snap = imgui.checkbox('Snap', self.snap)
        if imgui.is_item_hovered():
            imgui.set_tooltip('Keyframes snap to the playhead and (holding Shift) to beats.')
        imgui.same_line()
        imgui.text_disabled(f'{fmt_time(app.playhead)} / {fmt_time(total)}   '
                            'drag ruler: scrub | wheel: scroll | Ctrl+wheel: zoom | double-click lane: add keyframe | '
                            'right-click keyframe: menu')
        avail = imgui.get_content_region_avail()
        W, H = max(100.0, avail.x), max(RULER_H + KF_H + COLOR_H + TRACK_H + BEAT_H + VOWEL_H, avail.y)
        p0 = imgui.get_cursor_screen_pos()
        x0 = p0.x + 8
        usable = W - 16
        if self.fit_next:
            self.auto_fit = True
            self.fit_next = False
        if self.pps is None or self.auto_fit:
            self.pps = usable / span
            self.t0 = 0.0
        self.rect = (p0.x, p0.y, W, H)
        imgui.invisible_button('##timeline', V2(W, H))
        hovered = imgui.is_item_hovered()
        dl = imgui.get_window_draw_list()
        dl.push_clip_rect(V2(p0.x, p0.y), V2(p0.x + W, p0.y + H), True)
        dl.add_rect_filled(V2(p0.x, p0.y), V2(p0.x + W, p0.y + H), col32(0.09, 0.09, 0.11))
        y = p0.y
        y_ruler = y
        y_kf = y_ruler + RULER_H
        y_col = y_kf + KF_H
        y_trk = y_col + COLOR_H
        y_beat = y_trk + TRACK_H
        y_vow = y_beat + BEAT_H
        # ruler
        dl.add_rect_filled(V2(p0.x, y_ruler), V2(p0.x + W, y_ruler + RULER_H), col32(0.14, 0.14, 0.17))
        step = next((s for s in STEPS if s * self.pps >= 70), STEPS[-1])
        t_first = np.floor(self.t0 / step) * step
        t_last = self.t_of(p0.x + W, x0)
        t = t_first
        while t <= t_last:
            xx = self.x_of(t, x0)
            dl.add_line(V2(xx, y_ruler + 12), V2(xx, y_ruler + RULER_H), col32(0.5, 0.5, 0.55))
            dl.add_text(V2(xx + 3, y_ruler + 2), col32(0.75, 0.75, 0.8), fmt_time(t, step < 1))
            sub = step / 5
            for j in range(1, 5):
                sx = self.x_of(t + j * sub, x0)
                dl.add_line(V2(sx, y_ruler + 17), V2(sx, y_ruler + RULER_H), col32(0.35, 0.35, 0.4))
            t += step
        # tracks, waveform, beats, vowels
        offs, _ = eng.offsets()
        tracks = eng.tracks()
        for i, (o, ta) in enumerate(zip(offs, tracks)):
            if ta.duration <= 0:
                continue
            xa, xb = self.x_of(o, x0), self.x_of(o + ta.duration, x0)
            if xb < p0.x or xa > p0.x + W:
                continue
            base = (0.18, 0.22, 0.32) if i % 2 == 0 else (0.22, 0.18, 0.30)
            dl.add_rect_filled(V2(xa, y_trk), V2(xb, y_trk + TRACK_H), col32(*base))
            dl.add_line(V2(xa, y_trk), V2(xa, y_vow + VOWEL_H), col32(0.9, 0.9, 1.0, 0.6), 1.5)
            env = ta.envelope
            if env is not None and len(env):
                cx0, cx1 = max(xa, p0.x), min(xb, p0.x + W)
                ncol = int((cx1 - cx0) / 2)
                if ncol > 0:
                    ts = (np.arange(ncol) * 2 + cx0 - x0) / self.pps + self.t0 - o
                    te = ts + 2 / self.pps
                    a = np.clip((ts * 200).astype(int), 0, len(env) - 1)
                    b = np.clip((te * 200).astype(int) + 1, 1, len(env))
                    mid = y_trk + TRACK_H / 2
                    for c in range(ncol):
                        v = float(env[a[c]:max(b[c], a[c] + 1)].max())
                        hh = v * (TRACK_H / 2 - 4)
                        xx = cx0 + c * 2
                        dl.add_line(V2(xx, mid - hh), V2(xx, mid + hh), col32(0.55, 0.7, 0.95, 0.8))
            label = f'{i + 1}. {ta.name}'
            if ta.status != 'ready':
                label += f'  [{ta.status}{" %d%%" % int(ta.progress * 100) if ta.status == "spectrum" else ""}]'
            if ta.error:
                label += f'  ERROR: {ta.error}'
            dl.add_text(V2(xa + 5, y_trk + 3), col32(1, 1, 1, 0.9), label)
            if ta.feats is not None:
                beats = ta.feats['beats']
                if len(beats) > 1 and np.median(np.diff(beats)) * self.pps > 4:
                    for bt in beats:
                        bx = self.x_of(o + bt, x0)
                        if p0.x <= bx <= p0.x + W:
                            dl.add_line(V2(bx, y_beat + 2), V2(bx, y_beat + BEAT_H - 1), col32(1.0, 0.8, 0.3, 0.8))
            if ta.vowel is not None:
                cx0, cx1 = max(xa, p0.x), min(xb, p0.x + W)
                for xx in range(int(cx0), int(cx1), 2):
                    tt = self.t_of(xx, x0) - o
                    fi = int(min(max(tt * FEAT_FPS, 0), len(ta.vowel['voiced']) - 1))
                    v = float(ta.vowel['voiced'][fi])
                    if v < 0.2:
                        continue
                    c = ta.vowel['weights'][fi] @ VOWEL_COLS
                    dl.add_rect_filled(V2(xx, y_vow + 1), V2(xx + 2, y_vow + VOWEL_H - 1), col32(*c, min(1.0, v)))
        # colour lane
        ks = app.project.sorted_keys()
        sw = {k.id: keyframe_swatch(k) for k in ks}
        for xx in range(int(max(x0, p0.x)), int(min(self.x_of(span, x0), p0.x + W)), 3):
            tt = self.t_of(xx, x0)
            k0, k1, u = app.project.bracket(tt)
            c0 = sw[k0.id]
            cols = c0 if k1 is None else None
            if cols is None:
                c1 = sw[k1.id]
                n = max(len(c0), len(c1))
                c0r = c0[np.linspace(0, len(c0) - 1, n).astype(int)]
                c1r = c1[np.linspace(0, len(c1) - 1, n).astype(int)]
                cols = c0r * (1 - u) + c1r * u
            hh = COLOR_H / len(cols)
            for j, c in enumerate(cols):
                dl.add_rect_filled(V2(xx, y_col + j * hh), V2(xx + 3, y_col + (j + 1) * hh), col32(*c))
        # keyframes
        dl.add_rect_filled(V2(p0.x, y_kf), V2(p0.x + W, y_kf + KF_H), col32(0.11, 0.11, 0.14))
        mouse = imgui.get_io().mouse_pos
        hover_kf = None
        cy = y_kf + KF_H / 2
        for k in ks:
            kx = self.x_of(k.time, x0)
            if abs(mouse.x - kx) < 8 and y_kf <= mouse.y <= y_kf + KF_H:
                hover_kf = k
            c = sw[k.id][len(sw[k.id]) // 2]
            sel = k.id == app.selected_kf
            r = 8 if sel else 6.5
            pts = [V2(kx, cy - r), V2(kx + r, cy), V2(kx, cy + r), V2(kx - r, cy)]
            dl.add_triangle_filled(pts[0], pts[1], pts[2], col32(*c))
            dl.add_triangle_filled(pts[0], pts[2], pts[3], col32(*c))
            oc = col32(1, 1, 1) if sel else col32(0.1, 0.1, 0.1)
            for a_, b_ in zip(pts, pts[1:] + pts[:1]):
                dl.add_line(a_, b_, oc, 2.0 if sel else 1.0)
            if k.label and self.pps > 20:
                dl.add_text(V2(kx + 10, y_kf + 5), col32(0.9, 0.9, 0.9, 0.8), k.label)
        # playhead
        px = self.x_of(app.playhead, x0)
        dl.add_line(V2(px, p0.y), V2(px, p0.y + H), col32(1.0, 0.3, 0.3), 2.0)
        dl.add_triangle_filled(V2(px - 6, p0.y), V2(px + 6, p0.y), V2(px, p0.y + 9), col32(1.0, 0.3, 0.3))
        dl.pop_clip_rect()
        for name, yy, hh in (('keys', y_kf, KF_H), ('colour', y_col, COLOR_H), ('beats', y_beat, BEAT_H),
                             ('vowels', y_vow, VOWEL_H)):
            dl.add_text(V2(p0.x + W - 52, yy + max(0, (hh - 13) / 2)), col32(0.55, 0.55, 0.6, 0.7), name)
        # ---------------------------------------------------------------- interaction
        io = imgui.get_io()
        mt = self.t_of(mouse.x, x0)
        in_kf_lane = y_kf <= mouse.y <= y_kf + KF_H
        if hovered and imgui.is_mouse_double_clicked(0) and in_kf_lane and hover_kf is None:
            app.add_keyframe(self.snap_time(max(0.0, mt), io.key_shift, x0))
        elif imgui.is_item_clicked(0):
            if hover_kf is not None:
                app.select_kf(hover_kf.id)
                self.drag_kf = hover_kf.id
                self.drag_orig = hover_kf.time
                self.drag_mx = mouse.x
            else:
                self.scrubbing = True
        if imgui.is_item_clicked(1) and hover_kf is not None:
            self.ctx_kf = hover_kf.id
            app.select_kf(hover_kf.id)
            imgui.open_popup('kf_ctx')
        if self.drag_kf is not None:
            if imgui.is_mouse_down(0):
                k = app.project.key_by_id(self.drag_kf)
                if k is not None and abs(mouse.x - self.drag_mx) > 2:
                    nt = self.snap_time(max(0.0, self.drag_orig + (mouse.x - self.drag_mx) / self.pps), io.key_shift, x0)
                    if abs(nt - k.time) > 1e-6:
                        k.time = nt
                        app.touch()
            else:
                self.drag_kf = None
        if self.scrubbing:
            if imgui.is_mouse_down(0):
                app.seek(float(np.clip(mt, 0, total)))
            else:
                self.scrubbing = False
        if hovered and hover_kf is not None and self.drag_kf is None:
            from ..modes import MODES
            imgui.set_tooltip(f'{hover_kf.label or MODES[hover_kf.mode].name}\n{fmt_time(hover_kf.time)}  '
                              f'easing: {hover_kf.easing}  transition: {hover_kf.transition}')
        if hovered and io.mouse_wheel != 0:
            self.auto_fit = False
            if io.key_ctrl:
                f = 1.25 ** io.mouse_wheel
                tm = mt
                self.pps = float(np.clip(self.pps * f, 2.0, 4000.0))
                self.t0 = tm - (mouse.x - x0) / self.pps
            else:
                self.t0 -= io.mouse_wheel * 0.15 * usable / self.pps
        if hovered and io.mouse_wheel_h != 0:
            self.auto_fit = False
            self.t0 -= io.mouse_wheel_h * 0.15 * usable / self.pps
        self.t0 = float(np.clip(self.t0, -2.0, max(span - 1.0, 0.0)))
        if app.player.playing and self.drag_kf is None:
            # follow playhead
            if not self.auto_fit and (px > p0.x + W - 30 or px < p0.x):
                self.t0 = app.playhead - 0.1 * usable / self.pps
        if imgui.begin_popup('kf_ctx'):
            k = app.project.key_by_id(self.ctx_kf)
            if k is not None:
                imgui.text(f'Keyframe at {fmt_time(k.time)}')
                imgui.separator()
                if imgui.menu_item('Jump to', '', False)[0]:
                    app.seek(k.time)
                if imgui.menu_item('Duplicate at playhead', 'Ctrl+D', False)[0]:
                    app.duplicate_keyframe(k.id, app.playhead)
                if imgui.begin_menu('Easing to next'):
                    for e in EASINGS:
                        if imgui.menu_item(e, '', k.easing == e)[0]:
                            k.easing = e
                            app.touch()
                    imgui.end_menu()
                if imgui.begin_menu('Transition'):
                    for tr in ('morph', 'crossfade'):
                        if imgui.menu_item(tr, '', k.transition == tr)[0]:
                            k.transition = tr
                            app.touch()
                    imgui.end_menu()
                imgui.separator()
                if imgui.menu_item('Delete', 'Del', False, len(app.project.keyframes) > 1)[0]:
                    app.delete_keyframe(k.id)
            imgui.end_popup()

    def snap_time(self, t, shift, x0):
        if not self.snap:
            return t
        app = self.app
        if abs((t - app.playhead) * self.pps) < 6:
            return app.playhead
        if shift:
            best, bd = t, 1e9
            for o, ta in zip(app.engine.offsets()[0], app.engine.tracks()):
                if ta.feats is None:
                    continue
                b = ta.feats['beats'] + o
                if len(b):
                    j = int(np.argmin(np.abs(b - t)))
                    if abs(b[j] - t) < bd:
                        best, bd = float(b[j]), abs(b[j] - t)
            if bd * self.pps < 25:
                return best
        return t
