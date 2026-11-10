"""Reusable ImGui widgets: palette gradient editor, auto-generated parameter controls, swatches."""
import numpy as np
from imgui_bundle import imgui

from ..colorspace import hsv_to_rgb
from ..modes import MODES
from ..palettes import PRESETS, Palette, load_user_presets, random_palette, save_user_preset
from ..vowel import VOWELS

V2 = imgui.ImVec2


def col32(r, g, b, a=1.0):
    return imgui.get_color_u32(imgui.ImVec4(float(r), float(g), float(b), float(a)))


def tip(text):
    if text:
        imgui.set_item_tooltip(text)


def help_marker(text):
    imgui.same_line()
    imgui.text_disabled('(?)')
    tip(text)


class PaletteEditor:
    """Gradient bar with draggable stops. Double-click the bar to add a stop, right-click a stop to delete."""

    def __init__(self):
        self.sel = 0
        self.drag = None
        self.save_name = ''
        self.user = None
        self.seed = 1

    def draw(self, pal: Palette, uid='pal'):
        changed = False
        imgui.push_id(uid)
        if self.user is None:
            self.user = load_user_presets()
        avail = imgui.get_content_region_avail().x
        imgui.set_next_item_width(avail * 0.55)
        if imgui.begin_combo('##preset', pal.name):
            imgui.text_disabled('Built-in')
            for name in PRESETS:
                if imgui.selectable(name, name == pal.name)[0]:
                    self._assign(pal, Palette.preset(name))
                    changed = True
            if self.user:
                imgui.separator()
                imgui.text_disabled('Saved')
                for name, up in self.user.items():
                    if imgui.selectable(name + '##user', name == pal.name)[0]:
                        self._assign(pal, up.copy())
                        changed = True
            imgui.end_combo()
        tip('Palette presets (built-in and your saved ones).')
        imgui.same_line()
        if imgui.button('Random'):
            self.seed += 1
            self._assign(pal, random_palette(np.random.randint(1 << 30)))
            changed = True
        tip('Generate a harmonious random palette (OKLCh).')
        imgui.same_line()
        if imgui.button('Reverse'):
            self._assign(pal, pal.reversed())
            changed = True
        imgui.same_line()
        c, v = imgui.checkbox('Cyclic', pal.cyclic)
        tip('Cyclic palettes wrap around (needed for hue-like modes such as notes or angles).')
        if c:
            pal.cyclic = v
            changed = True
        # gradient bar
        w = max(60.0, imgui.get_content_region_avail().x)
        bar_h, mark_h = 26.0, 14.0
        p0 = imgui.get_cursor_screen_pos()
        imgui.invisible_button('##bar', V2(w, bar_h + mark_h))
        hovered = imgui.is_item_hovered()
        active = imgui.is_item_active()
        dl = imgui.get_window_draw_list()
        lut = pal.lut()
        seg = 96
        for i in range(seg):
            c0 = lut[int(i / seg * (len(lut) - 1))]
            c1 = lut[int((i + 1) / seg * (len(lut) - 1))]
            x0 = p0.x + w * i / seg
            x1 = p0.x + w * (i + 1) / seg
            dl.add_rect_filled_multi_color(V2(x0, p0.y), V2(x1 + 0.5, p0.y + bar_h), col32(*c0), col32(*c1),
                                           col32(*c1), col32(*c0))
        dl.add_rect(V2(p0.x, p0.y), V2(p0.x + w, p0.y + bar_h), col32(0.5, 0.5, 0.5), 2.0)
        self.sel = min(self.sel, len(pal.stops) - 1)
        for i, (pos, c) in enumerate(pal.stops):
            mx = p0.x + pos * w
            y0 = p0.y + bar_h
            dl.add_triangle_filled(V2(mx, y0), V2(mx - 6, y0 + mark_h - 1), V2(mx + 6, y0 + mark_h - 1), col32(*c))
            oc = col32(1, 1, 1) if i == self.sel else col32(0.25, 0.25, 0.25)
            dl.add_line(V2(mx, y0), V2(mx - 6, y0 + mark_h - 1), oc, 2.0 if i == self.sel else 1.0)
            dl.add_line(V2(mx - 6, y0 + mark_h - 1), V2(mx + 6, y0 + mark_h - 1), oc, 2.0 if i == self.sel else 1.0)
            dl.add_line(V2(mx + 6, y0 + mark_h - 1), V2(mx, y0), oc, 2.0 if i == self.sel else 1.0)
        mouse = imgui.get_io().mouse_pos
        mpos = float(np.clip((mouse.x - p0.x) / w, 0, 1))

        def nearest_stop():
            best, bd = None, 9.0
            for i, (pos, _c) in enumerate(pal.stops):
                d = abs(p0.x + pos * w - mouse.x)
                if d < bd:
                    best, bd = i, d
            return best
        if hovered and imgui.is_mouse_double_clicked(0):
            col = [float(x) for x in pal.sample(np.array([mpos]))[0]]
            pal.stops.append((mpos, col))
            self.sel = len(pal.stops) - 1
            pal.name = 'Custom'
            changed = True
        elif imgui.is_item_clicked(0):
            ns = nearest_stop()
            if ns is not None:
                self.sel = ns
                self.drag = ns
        if imgui.is_item_clicked(1):
            ns = nearest_stop()
            if ns is not None and len(pal.stops) > 2:
                pal.stops.pop(ns)
                self.sel = 0
                pal.name = 'Custom'
                changed = True
        if self.drag is not None and active and imgui.is_mouse_dragging(0, 1.0) and self.drag < len(pal.stops):
            pal.stops[self.drag] = (mpos, pal.stops[self.drag][1])
            pal.name = 'Custom'
            changed = True
        if not imgui.is_mouse_down(0):
            self.drag = None
        tip('Drag stops to move them. Double-click the bar to add a stop, right-click a stop to delete it.')
        # selected stop
        if pal.stops:
            pos, c = pal.stops[self.sel]
            imgui.set_next_item_width(avail * 0.5)
            ch, nc = imgui.color_edit3(f'Stop {self.sel + 1}##stopcol', list(c))
            if ch:
                pal.stops[self.sel] = (pos, [float(x) for x in nc])
                pal.name = 'Custom'
                changed = True
            imgui.same_line()
            imgui.set_next_item_width(max(60.0, imgui.get_content_region_avail().x - 70))
            ch, npos = imgui.slider_float('##stoppos', float(pos), 0.0, 1.0, 'pos %.3f')
            if ch:
                pal.stops[self.sel] = (npos, pal.stops[self.sel][1])
                pal.name = 'Custom'
                changed = True
            imgui.same_line()
            imgui.begin_disabled(len(pal.stops) <= 2)
            if imgui.button('Del##stop'):
                pal.stops.pop(self.sel)
                self.sel = 0
                pal.name = 'Custom'
                changed = True
            imgui.end_disabled()
        imgui.set_next_item_width(avail * 0.55)
        _c, self.save_name = imgui.input_text('##savename', self.save_name)
        imgui.same_line()
        imgui.begin_disabled(not self.save_name.strip())
        if imgui.button('Save as preset'):
            save_user_preset(pal, self.save_name.strip())
            pal.name = self.save_name.strip()
            self.user = load_user_presets()
            self.save_name = ''
        imgui.end_disabled()
        imgui.pop_id()
        return changed

    @staticmethod
    def _assign(dst, src):
        dst.stops = [(p, list(c)) for p, c in src.stops]
        dst.cyclic = src.cyclic
        dst.name = src.name


def param_widget(prm, value, app=None, kf=None):
    """Draws one parameter control. Returns (changed, new_value)."""
    label = f'{prm.label}##{prm.key}'
    if prm.kind == 'float':
        flags = imgui.SliderFlags_.logarithmic.value if prm.log else 0
        ch, v = imgui.slider_float(label, float(value), float(prm.lo), float(prm.hi), '%.3f', flags)
        tip(prm.tip)
        return ch, float(v)
    if prm.kind == 'int':
        ch, v = imgui.slider_int(label, int(value), int(prm.lo), int(prm.hi))
        tip(prm.tip)
        return ch, int(v)
    if prm.kind == 'bool':
        ch, v = imgui.checkbox(label, bool(value))
        tip(prm.tip)
        return ch, bool(v)
    if prm.kind == 'choice':
        idx = prm.choices.index(value) if value in prm.choices else 0
        ch, i = imgui.combo(label, idx, prm.choices)
        tip(prm.tip)
        return ch, prm.choices[i]
    if prm.kind == 'color':
        ch, v = imgui.color_edit3(label, list(value))
        tip(prm.tip)
        return ch, [float(x) for x in v]
    if prm.kind == 'zones':
        return zones_widget(value, app, kf)
    imgui.text(f'{prm.label}: {value}')
    return False, value


def zones_widget(zones, app, kf):
    zones = [dict(z) for z in (zones or [])]
    changed = False
    imgui.text('Detector zones')
    help_marker('Each zone counts the visible dots inside it; above the threshold it triggers and flashes.')
    remove = None
    for i, z in enumerate(zones):
        imgui.push_id(f'zone{i}')
        open_ = imgui.tree_node_ex(f'Zone {i + 1}', imgui.TreeNodeFlags_.default_open.value)
        imgui.same_line()
        if imgui.small_button('x'):
            remove = i
        if open_:
            for key, lo, hi in (('x', -1.0, 1.0), ('y', 0.0, 1.0), ('r', 0.01, 0.5)):
                ch, v = imgui.slider_float(key, float(z[key]), lo, hi)
                if ch:
                    z[key] = float(v)
                    changed = True
            ch, v = imgui.slider_int('threshold', int(z['threshold']), 1, 200)
            tip('Trigger when more than this many dots are inside.')
            if ch:
                z['threshold'] = int(v)
                changed = True
            ch, v = imgui.color_edit3('colour', list(z['color']))
            if ch:
                z['color'] = [float(c) for c in v]
                changed = True
            imgui.tree_pop()
        imgui.pop_id()
    if remove is not None:
        zones.pop(remove)
        changed = True
    if imgui.button('Add zone'):
        zones.append({'x': 0.0, 'y': 0.5, 'r': 0.08, 'threshold': 12, 'color': [1.0, 1.0, 1.0]})
        changed = True
    if app is not None and kf is not None:
        imgui.same_line()
        armed = app.place_mode is not None and app.place_mode[0] == 'zone'
        if imgui.button('Stop placing' if armed else 'Place zone in viewer'):
            app.place_mode = None if armed else ('zone', kf.id)
        tip('Then click in the viewer to drop a new zone there.')
    return changed, zones


def keyframe_swatch(k, n=6):
    """n representative sRGB colours for a keyframe (for the timeline)."""
    m = MODES[k.mode]
    if m.native and not k.use_palette:
        p = k.params
        if k.mode == 'vowel':
            return np.array([p[v] for v in VOWELS], np.float32)
        if k.mode == 'stems':
            return np.array([p['vocals'], p['drums'], p['bass'], p['other']], np.float32)
        if k.mode == 'solid':
            return np.array([p['color']] * 2, np.float32)
        if k.mode in ('mono',):
            return np.ones((2, 3), np.float32)
        return hsv_to_rgb(np.arange(n) / n)
    lut = k.palette.lut()
    return lut[np.linspace(0, len(lut) - 1, n).astype(int)]
