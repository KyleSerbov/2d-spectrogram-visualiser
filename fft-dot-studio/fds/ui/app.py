"""FFT Dot Studio editor application (Dear ImGui via imgui-bundle / hello_imgui, ModernGL rendering)."""
import json
import os
import time
import traceback

import numpy as np
from imgui_bundle import hello_imgui, imgui

from .. import analysis as A
from ..analysis import ANALYSIS_PRESETS, WINDOW_FUNCS, WINDOW_SIZES, AnalysisManager, sanitize_settings
from ..audio_io import DIALOG_FILTER
from ..engine import Engine
from ..export import AUDIO_CODECS, CODECS, FPS_CHOICES, RESOLUTIONS, Exporter
from ..modes import CATEGORIES, MODE_ORDER, MODES
from ..playback import Player
from ..project import (ANIMATABLE, EASINGS, RENDER_CHOICES, RENDER_DEFAULTS, RENDER_PRESETS, RENDER_TIPS, Keyframe,
                       Project, y_to_freq)
from ..vowel import DEFAULT_VOWEL_COLORS, REF_FORMANTS, VOWELS
from .timeline import Timeline, fmt_time
from .widgets import V2, PaletteEditor, col32, help_marker, param_widget, tip

PREVIEW_FPS = [('30', 30), ('60', 60), ('120', 120), ('uncapped', 0)]
PROJECT_FILTER = ['FFT Dot Studio project', '*.fdsproj *.json', 'All files', '*']
RENDER_GROUPS = [
    ('Frequency axis', ['freq_min', 'freq_max', 'freq_scale']),
    ('Amplitude', ['amp_scale', 'db_floor', 'normalise', 'running_release_s', 'threshold', 'gamma', 'brightness']),
    ('Dot selection', ['peak_pick', 'peak_min_dist', 'max_dots']),
    ('Time', ['smooth_attack_ms', 'smooth_release_ms', 'trail_ms']),
    ('Dots & look', ['dot_size', 'dot_shape', 'glow', 'glow_radius', 'tonemap', 'saturation', 'background']),
    ('Geometry', ['pan_mode', 'pan_spread', 'flip_x', 'flip_y', 'mirror_x', 'mirror_y', 'edge_margin']),
]
RENDER_RANGES = {
    'freq_min': (5.0, 2000.0, True), 'freq_max': (1000.0, 40000.0, True), 'db_floor': (-120.0, -12.0, False),
    'running_release_s': (0.1, 20.0, True), 'threshold': (0.0001, 0.5, True), 'gamma': (0.2, 4.0, False),
    'brightness': (0.05, 300.0, True), 'peak_min_dist': (0.001, 0.2, True), 'max_dots': (0, 5000, False),
    'smooth_attack_ms': (0.0, 500.0, False), 'smooth_release_ms': (0.0, 2000.0, False), 'trail_ms': (0.0, 2000.0, False),
    'dot_size': (1.0, 40.0, False), 'glow': (0.0, 3.0, False), 'glow_radius': (0.3, 4.0, False),
    'saturation': (0.0, 2.0, False), 'pan_spread': (0.1, 3.0, False), 'edge_margin': (0.0, 0.2, False),
}
ANALYSIS_TIPS = {
    'window': 'FFT window size. Larger = finer pitch detail but blurrier in time.',
    'hop': 'Samples between analysis frames (window / overlap). Smaller = smoother motion and no missed beats.',
    'window_fn': 'Window function: Hann (original), Blackman-Harris (cleaner peaks), Gaussian, Hamming.',
    'pad': 'Zero-padding factor: interpolates the spectrum (more, smoother dots) without changing resolution.',
    'multires': 'Multi-resolution: a large window for bass, the main window for mids and a small one for treble, '
                'crossfaded at the crossovers.',
    'bass_window': 'Window size used below the low crossover.',
    'treble_window': 'Window size used above the high crossover.',
    'cross_low': 'Bass / mid crossover frequency.', 'cross_high': 'Mid / treble crossover frequency.',
    'blend_oct': 'Width of the crossfade around each crossover, in octaves.',
}


class History:
    def __init__(self, cap=200):
        self.undo, self.redo, self.last, self.cap = [], [], None, cap

    def snapshot(self, d):
        s = json.dumps(d, sort_keys=True)
        if s == self.last:
            return
        if self.last is not None:
            self.undo.append(self.last)
            self.undo = self.undo[-self.cap:]
            self.redo.clear()
        self.last = s

    def do_undo(self):
        if not self.undo:
            return None
        self.redo.append(self.last)
        self.last = self.undo.pop()
        return json.loads(self.last)

    def do_redo(self):
        if not self.redo:
            return None
        self.undo.append(self.last)
        self.last = self.redo.pop()
        return json.loads(self.last)


class StudioApp:
    def __init__(self, project_path=None, tracks=None, gl=None, headless=False):
        self.project = Project.load(project_path) if project_path else Project()
        if tracks:
            self.project.tracks += [os.path.abspath(t) for t in tracks]
        self.manager = AnalysisManager(use_demucs=self.project.use_demucs, vocal_source=self.project.vocal_source)
        self.manager.set_settings(self.project.analysis)
        self.engine = Engine(self.project, self.manager)
        for t in self.project.tracks:
            self.manager.get(t)
        self.player = Player(self.engine)
        self.timeline = Timeline(self)
        self.paleditor = PaletteEditor()
        self.playhead = 0.0
        self.selected_kf = self.project.sorted_keys()[0].id
        self.place_mode = None
        self.history = History()
        self.history.snapshot(self.project.to_dict())
        self.gl = gl
        self.renderer = None
        self.last_key = None
        self.last_dots = None
        self.dialog = None
        self.exporter = None
        self.export_msg = ''
        self.messages = []
        self.fullscreen = False
        self.headless = headless
        self.frame_times = []
        self.last_render_ms = 0.0
        self.view_rect = None
        self.export_res_name = '1920x1080 (1080p)'
        self.show_help = False

    # ================================================================== state helpers
    def touch(self):
        self.project.revision += 1

    def log(self, msg):
        self.messages.append((time.time(), msg))
        self.messages = self.messages[-6:]

    def total(self):
        return self.engine.total_duration()

    def seek(self, t):
        self.playhead = float(np.clip(t, 0.0, max(self.total(), 0.0)))
        self.player.seek(self.playhead)

    def toggle_play(self):
        if self.player.playing:
            self.playhead = self.player.stop()
        else:
            if self.total() <= 0:
                self.log('Add some audio first (File > Add audio, or Ctrl+I).')
                return
            if self.playhead >= self.total() - 0.05:
                self.playhead = 0.0
            self.player.play(self.playhead)
            if not self.player.device_ok:
                self.log('No audio output device: playing silently on the wall clock.')

    def view_time(self):
        """Timeline time the viewer shows. While playing: the audio clock minus output latency."""
        if self.player.playing:
            lat = self.project.preview_latency
            t = self.player.time() if lat.get('mode', 'measured') == 'measured' else self.player.submitted_time()
            return t - float(lat.get('ms', 0.0)) / 1000.0
        return self.playhead

    def current_kf(self):
        k = self.project.key_by_id(self.selected_kf)
        if k is None:
            k = self.project.bracket(self.playhead)[0]
            self.selected_kf = k.id
        return k

    def select_kf(self, kid):
        self.selected_kf = kid

    def add_keyframe(self, t):
        for k in self.project.keyframes:
            if abs(k.time - t) < 1e-3:
                self.selected_kf = k.id
                return k
        base = self.project.bracket(t)[0]
        k = base.copy()
        k.time = float(t)
        k.label = ''
        self.project.keyframes.append(k)
        self.selected_kf = k.id
        self.touch()
        return k

    def duplicate_keyframe(self, kid, t):
        src = self.project.key_by_id(kid)
        if src is None:
            return
        k = src.copy()
        k.time = float(t) if abs(t - src.time) > 1e-3 else src.time + 1.0
        self.project.keyframes.append(k)
        self.selected_kf = k.id
        self.touch()

    def delete_keyframe(self, kid):
        if len(self.project.keyframes) <= 1:
            self.log('A project needs at least one keyframe.')
            return
        self.project.keyframes = [k for k in self.project.keyframes if k.id != kid]
        self.selected_kf = self.project.bracket(self.playhead)[0].id
        self.touch()

    def jump_keyframe(self, direction):
        ks = self.project.sorted_keys()
        if direction > 0:
            nxt = [k for k in ks if k.time > self.playhead + 1e-4]
            k = nxt[0] if nxt else ks[-1]
        else:
            prv = [k for k in ks if k.time < self.playhead - 1e-4]
            k = prv[-1] if prv else ks[0]
        self.seek(k.time)
        self.selected_kf = k.id

    def add_tracks(self, paths):
        for p in paths:
            p = os.path.abspath(p)
            self.project.tracks.append(p)
            self.manager.get(p)
            self.log(f'Added {os.path.basename(p)}')
        self.timeline.fit_next = True
        self.touch()

    def remove_track(self, i):
        self.player.stop()
        del self.project.tracks[i]
        self.timeline.fit_next = True
        self.touch()

    def move_track(self, i, d):
        j = i + d
        if 0 <= j < len(self.project.tracks):
            tr = self.project.tracks
            tr[i], tr[j] = tr[j], tr[i]
            self.touch()

    def set_project(self, proj):
        self.player.stop()
        self.project = proj
        self.engine.project = proj
        self.engine.invalidate()
        self.manager.use_demucs = proj.use_demucs
        self.manager.vocal_source = proj.vocal_source
        self.manager.set_settings(proj.analysis)
        for t in proj.tracks:
            self.manager.get(t)
        self.selected_kf = proj.sorted_keys()[0].id
        self.playhead = 0.0
        self.history = History()
        self.history.snapshot(proj.to_dict())
        self.timeline.fit_next = True
        self.touch()

    def new_project(self):
        self.set_project(Project())

    def open_project(self, path):
        try:
            self.set_project(Project.load(path))
            self.log(f'Opened {os.path.basename(path)}')
        except Exception as e:
            self.log(f'Could not open project: {e}')

    def save_project(self, path=None):
        path = path or self.project.path
        if not path:
            self.open_dialog('save_project')
            return
        if not os.path.splitext(path)[1]:
            path += '.fdsproj'
        try:
            self.project.save(path)
            self.log(f'Saved {os.path.basename(path)}')
        except Exception as e:
            self.log(f'Save failed: {e}')

    def apply_history(self, d):
        if d is None:
            return
        path = self.project.path
        proj = Project.from_dict(d)
        proj.path = path
        proj.revision = self.project.revision + 1
        self.project.__dict__.update(proj.__dict__)
        self.manager.set_settings(self.project.analysis)
        for t in self.project.tracks:
            self.manager.get(t)
        if self.project.key_by_id(self.selected_kf) is None:
            self.selected_kf = self.project.sorted_keys()[0].id

    def undo(self):
        self.apply_history(self.history.do_undo())
        self._hist_rev = self.project.revision

    def redo(self):
        self.apply_history(self.history.do_redo())
        self._hist_rev = self.project.revision

    # ================================================================== dialogs
    def open_dialog(self, kind):
        if self.headless:
            return
        from imgui_bundle import portable_file_dialogs as pfd
        start = os.path.dirname(self.project.path) if self.project.path else os.path.expanduser('~')
        if kind == 'add_audio':
            d = pfd.open_file('Add audio (one or more files)', start, DIALOG_FILTER, pfd.opt.multiselect)
        elif kind == 'open_project':
            d = pfd.open_file('Open project', start, PROJECT_FILTER)
        elif kind == 'save_project':
            d = pfd.save_file('Save project', os.path.join(start, 'untitled.fdsproj'), PROJECT_FILTER)
        elif kind == 'export_path':
            default = self.project.export.get('path') or os.path.join(start, 'fft-dots.mp4')
            d = pfd.save_file('Export video', default, ['MP4 video', '*.mp4', 'Matroska', '*.mkv', 'QuickTime', '*.mov'])
        else:
            return
        self.dialog = (kind, d)

    def poll_dialog(self):
        if self.dialog is None:
            return
        kind, d = self.dialog
        if not d.ready(0):
            return
        self.dialog = None
        res = d.result()
        if not res:
            return
        if kind == 'add_audio':
            self.add_tracks(res)
        elif kind == 'open_project':
            self.open_project(res[0])
        elif kind == 'save_project':
            self.save_project(res)
        elif kind == 'export_path':
            path = res if os.path.splitext(res)[1] else res + '.mp4'
            self.project.export['path'] = path
            self.touch()

    # ================================================================== export
    def start_export(self, path=None):
        ex = self.project.export
        path = path or ex.get('path')
        if not path:
            self.open_dialog('export_path')
            return
        blockers = self.engine.export_blockers()
        if blockers:
            self.export_msg = 'Waiting for analysis the colour modes need: ' + '; '.join(blockers)
            self.log(self.export_msg)
            return
        self.player.stop()
        try:
            self.exporter = Exporter(self.engine, self.gl, path, ex['width'], ex['height'], ex['fps'], ex['codec'],
                                     ex['audio'], ex['quality'])
            self.exporter.start()
            self.export_msg = ''
        except Exception as e:
            self.exporter = None
            self.export_msg = f'Export failed: {e}'
            self.log(self.export_msg)

    # ================================================================== frame
    def post_init(self):
        import moderngl
        from ..renderer import Renderer
        if self.gl is None:
            self.gl = moderngl.create_context()
        self.renderer = Renderer(self.gl, 640, 360)
        io = imgui.get_io()
        io.config_windows_move_from_title_bar_only = True

    def before_exit(self):
        self.player.stop()
        if self.exporter is not None and not self.exporter.done:
            self.exporter.cancel()

    def pre_frame(self):
        if self.player.playing:
            self.playhead = self.player.time()
            if self.playhead >= self.total() or self.player.ended:
                self.playhead = self.player.stop()
                self.playhead = min(self.playhead, self.total())
        self.poll_dialog()
        if self.exporter is not None and not self.exporter.done and not self.exporter.cancelled:
            try:
                self.exporter.step(0.03)
            finally:
                if not self.headless:
                    self.gl.screen.use()
        # undo snapshots on settled edits
        if getattr(self, '_hist_rev', None) != self.project.revision and not imgui.is_mouse_down(0):
            self.history.snapshot(self.project.to_dict())
            self._hist_rev = self.project.revision
        if not self.headless:
            rp = hello_imgui.get_runner_params()
            fps = self.project.preview_fps
            busy = self.player.playing or self.exporter is not None or self.manager.busy()
            rp.fps_idling.enable_idling = not busy
            rp.fps_idling.fps_max = int(fps) if fps else 0
            rp.fps_idling.vsync_to_monitor = bool(fps)
        now = time.perf_counter()
        self.frame_times = [t for t in self.frame_times if now - t < 1.0] + [now]

    def shortcuts(self):
        io = imgui.get_io()
        if io.want_text_input:
            return
        K = imgui.Key
        ctrl, shift = io.key_ctrl, io.key_shift
        pressed = lambda k, rep=False: imgui.is_key_pressed(k, rep)
        if pressed(K.space):
            self.toggle_play()
        if self.fullscreen and (pressed(K.escape) or pressed(K.f)):
            self.fullscreen = False
        elif pressed(K.f) and not ctrl:
            self.fullscreen = True
        if ctrl and pressed(K.z):
            self.redo() if shift else self.undo()
        elif ctrl and pressed(K.y):
            self.redo()
        elif ctrl and pressed(K.s):
            self.open_dialog('save_project') if shift else self.save_project()
        elif ctrl and pressed(K.o):
            self.open_dialog('open_project')
        elif ctrl and pressed(K.n):
            self.new_project()
        elif ctrl and pressed(K.i):
            self.open_dialog('add_audio')
        elif ctrl and pressed(K.e):
            self.start_export()
        elif ctrl and pressed(K.d):
            self.duplicate_keyframe(self.selected_kf, self.playhead)
        elif pressed(K.k):
            self.add_keyframe(self.playhead)
        elif pressed(K.delete):
            self.delete_keyframe(self.selected_kf)
        elif pressed(K.m):
            self.player.muted = not self.player.muted
        elif pressed(K.home):
            self.seek(0.0)
        elif pressed(K.end):
            self.seek(self.total())
        if pressed(K.left_arrow, True):
            if ctrl:
                self.jump_keyframe(-1)
            else:
                self.seek(self.playhead - (1.0 if shift else 1.0 / 30))
        if pressed(K.right_arrow, True):
            if ctrl:
                self.jump_keyframe(1)
            else:
                self.seek(self.playhead + (1.0 if shift else 1.0 / 30))

    def render_view(self, w, h):
        if self.renderer is None:
            return
        w = int(np.clip(w, 32, 3840))
        h = int(np.clip(h, 32, 2160))
        tv = self.view_time()
        key = (self.project.revision, round(tv, 5), w, h, self.manager.revision,
               tuple(ta.spectrum_key for ta in self.engine.tracks()))
        if key == self.last_key:
            return
        t0 = time.perf_counter()
        try:
            self.renderer.resize(w, h)
            dots = self.engine.gather(tv, w / h)
            self.renderer.render(dots)
            self.last_dots = dots
        except Exception as e:
            self.log(f'Render error: {e}')
            traceback.print_exc()
        finally:
            if not self.headless:
                self.gl.screen.use()
        self.last_key = key
        self.last_render_ms = (time.perf_counter() - t0) * 1000

    # ================================================================== panels
    def gui_viewer(self):
        avail = imgui.get_content_region_avail()
        bar_h = imgui.get_frame_height_with_spacing() * 2 + 4
        W, H = max(64.0, avail.x), max(64.0, avail.y - bar_h)
        self._draw_view(W, H)
        self._transport()

    def _draw_view(self, W, H, hud=True):
        ex = self.project.export
        aspect = ex['width'] / max(ex['height'], 1)
        if W / H > aspect:
            vh, vw = H, H * aspect
        else:
            vw, vh = W, W / aspect
        self.render_view(vw, vh)
        cur = imgui.get_cursor_pos()
        imgui.set_cursor_pos(V2(cur.x + (W - vw) / 2, cur.y + (H - vh) / 2))
        p0 = imgui.get_cursor_screen_pos()
        if self.renderer is not None:
            imgui.image(imgui.ImTextureRef(self.renderer.texture.glo), V2(vw, vh), V2(0, 1), V2(1, 0))
        else:
            imgui.dummy(V2(vw, vh))
        self.view_rect = (p0.x, p0.y, vw, vh)
        if imgui.is_item_hovered() and imgui.is_item_clicked(0) and self.place_mode:
            m = imgui.get_io().mouse_pos
            x = (m.x - p0.x) / vw * 2 - 1
            y = 1 - (m.y - p0.y) / vh
            r = self.project.render
            if r['flip_x']:
                x = -x
            if r['flip_y']:
                y = 1 - y
            self.place_at(float(x), float(y))
        if self.place_mode and imgui.is_item_hovered():
            imgui.set_tooltip('Click to place ' + ('the centre' if self.place_mode[0] == 'centre' else 'a zone'))
        imgui.set_cursor_pos(V2(cur.x, cur.y + H))
        if hud:
            self._hud(p0, vw, vh)

    def _hud(self, p0, vw, vh):
        dl = imgui.get_window_draw_list()
        k0, k1, u = self.project.bracket(self.view_time())
        mode = MODES[k0.mode].name if (k1 is None or u <= 0) else f'{MODES[k0.mode].name}  ->  {MODES[k1.mode].name}  {u * 100:.0f}%'
        ti, lt = self.engine.locate(self.view_time())
        name = self.engine.tracks()[ti].name if ti is not None else '-'
        n = len(self.last_dots['x']) if self.last_dots else 0
        fps = len(self.frame_times)
        lines = [f'{fmt_time(self.playhead)}   {name}', mode, f'{n} dots   {fps} fps   render {self.last_render_ms:.1f} ms']
        for i, line in enumerate(lines):
            dl.add_text(V2(p0.x + 8, p0.y + 6 + 15 * i), col32(1, 1, 1, 0.55), line)
        busy = [ta for ta in self.engine.tracks() if ta.status != 'ready']
        if busy:
            ta = busy[0]
            dl.add_text(V2(p0.x + 8, p0.y + vh - 20), col32(1, 0.85, 0.4, 0.9),
                        f'Analysing {ta.name}: {ta.status} {int(ta.progress * 100) if ta.status == "spectrum" else ""}')
        if not self.engine.tracks():
            msg = 'Drop in some music: File > Add audio (Ctrl+I)'
            dl.add_text(V2(p0.x + vw / 2 - 150, p0.y + vh / 2), col32(1, 1, 1, 0.7), msg)

    def place_at(self, x, y):
        kind, kid = self.place_mode
        k = self.project.key_by_id(kid)
        if k is None:
            self.place_mode = None
            return
        if kind == 'centre':
            k.set_param('centre', 'custom')
            k.set_param('cx', x)
            k.set_param('cy', y)
            self.place_mode = None
        elif kind == 'zone':
            zones = list(k.params['zones'] or [])
            zones.append({'x': x, 'y': y, 'r': 0.08, 'threshold': 12, 'color': [1.0, 1.0, 1.0]})
            k.set_param('zones', zones)
        self.touch()

    def _transport(self):
        imgui.separator()
        if imgui.button('|<'):
            self.seek(0)
        tip('Start (Home)')
        imgui.same_line()
        if imgui.button('<K'):
            self.jump_keyframe(-1)
        tip('Previous keyframe (Ctrl+Left)')
        imgui.same_line()
        if imgui.button('Pause' if self.player.playing else ' Play ', V2(70, 0)):
            self.toggle_play()
        tip('Play / pause (Space)')
        imgui.same_line()
        if imgui.button('K>'):
            self.jump_keyframe(1)
        tip('Next keyframe (Ctrl+Right)')
        imgui.same_line()
        if imgui.button('>|'):
            self.seek(self.total())
        imgui.same_line()
        imgui.text(f'{fmt_time(self.playhead)} / {fmt_time(self.total())}')
        imgui.same_line()
        imgui.set_next_item_width(90)
        ch, v = imgui.slider_float('##vol', self.player.volume, 0.0, 1.5, 'vol %.2f')
        if ch:
            self.player.volume = v
        imgui.same_line()
        _c, self.player.muted = imgui.checkbox('Mute (M)', self.player.muted)
        imgui.same_line()
        lat = self.project.preview_latency
        modes = ['measured', 'manual']
        meas = self.player.latency() * 1000
        imgui.set_next_item_width(150)
        ch, i = imgui.combo('##latmode', modes.index(lat.get('mode', 'measured')),
                            [f'Output latency: device ({meas:.0f} ms)' if self.player.stream else 'Output latency: device',
                             'Output latency: manual only'])
        if ch:
            lat['mode'] = modes[i]
            self.touch()
        tip('PREVIEW ONLY - never affects export.\nCompensates the delay between audio leaving this program and '
            'reaching your ears (driver, DAC, Bluetooth...). "device" uses the latency reported by the audio '
            'stream; "manual" uses only the slider.')
        imgui.same_line()
        imgui.set_next_item_width(150)
        ch, v = imgui.slider_float('##latms', float(lat.get('ms', 0.0)), -200.0, 400.0, 'extra latency %.0f ms')
        if ch:
            lat['ms'] = float(v)
            self.touch()
        tip('PREVIEW ONLY. Extra output latency in ms: the dots are drawn for (audio clock - latency). Increase it '
            'if the dots appear before you hear the sound (e.g. Bluetooth headphones).')
        imgui.same_line()
        imgui.set_next_item_width(110)
        labels = [p[0] for p in PREVIEW_FPS]
        cur = next((i for i, p in enumerate(PREVIEW_FPS) if p[1] == self.project.preview_fps), 0)
        ch, i = imgui.combo('Preview FPS', cur, labels)
        if ch:
            self.project.preview_fps = PREVIEW_FPS[i][1]
            self.touch()
        tip('Frame rate cap for the live preview. Visuals always follow the audio clock, so they never drift.')
        if self.messages:
            age = time.time() - self.messages[-1][0]
            if age < 8:
                imgui.text_colored(imgui.ImVec4(1, 0.85, 0.4, 1 - age / 8), self.messages[-1][1])

    def gui_timeline(self):
        self.timeline.draw()

    # ------------------------------------------------------------------ keyframe inspector
    def gui_keyframe(self):
        proj = self.project
        k = self.current_kf()
        ks = proj.sorted_keys()
        idx = ks.index(k)
        imgui.text(f'Keyframe {idx + 1} / {len(ks)}')
        imgui.same_line()
        if imgui.small_button('<'):
            self.selected_kf = ks[max(0, idx - 1)].id
            self.seek(proj.key_by_id(self.selected_kf).time)
        imgui.same_line()
        if imgui.small_button('>'):
            self.selected_kf = ks[min(len(ks) - 1, idx + 1)].id
            self.seek(proj.key_by_id(self.selected_kf).time)
        imgui.same_line()
        if imgui.small_button('+ at playhead (K)'):
            k = self.add_keyframe(self.playhead)
        imgui.same_line()
        if imgui.small_button('Duplicate'):
            self.duplicate_keyframe(k.id, self.playhead)
        imgui.same_line()
        imgui.begin_disabled(len(ks) <= 1)
        if imgui.small_button('Delete'):
            self.delete_keyframe(k.id)
            imgui.end_disabled()
            return
        imgui.end_disabled()
        k0, k1, u = proj.bracket(self.playhead)
        if k1 is not None and 0 < u < 1:
            imgui.text_disabled(f'Playhead blend: {MODES[k0.mode].name} -> {MODES[k1.mode].name} ({u * 100:.0f}%)')
        imgui.separator()
        ch, v = imgui.drag_float('Time (s)', k.time, 0.01, 0.0, max(self.total(), 1e6), '%.3f')
        if ch:
            k.time = max(0.0, v)
            self.touch()
        ch, v = imgui.input_text('Label', k.label)
        if ch:
            k.label = v
            self.touch()
        ch, i = imgui.combo('Easing to next', EASINGS.index(k.easing) if k.easing in EASINGS else 0, EASINGS)
        if ch:
            k.easing = EASINGS[i]
            self.touch()
        tip('How the transition from this keyframe to the next one accelerates. "hold" = jump at the next keyframe.')
        trs = ['morph', 'crossfade']
        ch, i = imgui.combo('Transition', trs.index(k.transition), trs)
        if ch:
            k.transition = trs[i]
            self.touch()
        tip('morph: when both keyframes use the same mode, parameters and palettes are interpolated.\n'
            'crossfade: both modes are evaluated and every dot colour is blended in OKLab.')
        imgui.separator_text('Colour mode')
        m = MODES[k.mode]
        if imgui.begin_combo('Mode', m.name, imgui.ComboFlags_.height_largest.value):
            for cat in CATEGORIES:
                imgui.separator_text('Ported from earlier versions' if cat == 'Ported' else 'New modes')
                for key in MODE_ORDER:
                    mm = MODES[key]
                    if mm.category != cat:
                        continue
                    if imgui.selectable(mm.name, key == k.mode)[0]:
                        k.set_mode(key, adopt_palette=not k.use_palette)
                        self.touch()
                    tip(mm.desc)
            imgui.end_combo()
        m = MODES[k.mode]
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(m.desc)
        imgui.pop_text_wrap_pos()
        if 'stems' in m.needs and not any(ta.stems is not None for ta in self.engine.tracks()):
            imgui.text_colored(imgui.ImVec4(1, 0.7, 0.3, 1), 'Needs Demucs stems (see Analysis panel).')
        if 'vowel' in m.needs and not any(ta.vowel is not None for ta in self.engine.tracks()):
            imgui.text_colored(imgui.ImVec4(1, 0.7, 0.3, 1), 'Vowel analysis still running...')
        params = k.params
        changed = False
        imgui.push_item_width(-150)
        for prm in m.params:
            if prm.key in ('cx', 'cy') and params.get('centre') != 'custom':
                continue
            ch, v = param_widget(prm, params[prm.key], self, k)
            if ch:
                params[prm.key] = v
                changed = True
            if prm.key == 'centre':
                imgui.same_line()
                armed = self.place_mode is not None and self.place_mode[0] == 'centre'
                if imgui.small_button('cancel pick' if armed else 'pick in viewer'):
                    self.place_mode = None if armed else ('centre', k.id)
        imgui.pop_item_width()
        if changed:
            k.params_by_mode[k.mode] = params
            self.touch()
        if imgui.button('Reset mode parameters'):
            k.params_by_mode.pop(k.mode, None)
            self.touch()
        imgui.separator_text('Palette')
        if m.native:
            ch, v = imgui.checkbox('Use palette instead of native colours', k.use_palette)
            if ch:
                k.use_palette = v
                self.touch()
            tip('This mode has its own colours (notes / vowels / stems...). Tick to send its value through the palette.')
        if (not m.native) or k.use_palette:
            if self.paleditor.draw(k.palette, f'pal_{k.id}'):
                self.touch()
        else:
            imgui.text_disabled('Native colours in use.')
        imgui.separator_text('Animate render settings on this keyframe')
        imgui.push_item_width(-150)
        for key in ANIMATABLE:
            on = key in k.render_overrides
            ch, v = imgui.checkbox(f'##ov_{key}', on)
            tip('Override (and tween) this render setting at this keyframe.')
            if ch:
                if v:
                    k.render_overrides[key] = self.project.render[key]
                else:
                    k.render_overrides.pop(key, None)
                self.touch()
            imgui.same_line()
            imgui.begin_disabled(not on)
            val = k.render_overrides.get(key, self.project.render[key])
            ch, nv = self.render_widget(key, val)
            if ch and on:
                k.render_overrides[key] = nv
                self.touch()
            imgui.end_disabled()
        imgui.pop_item_width()

    def render_widget(self, key, val):
        label = key.replace('_', ' ')
        if key in RENDER_CHOICES:
            ch_list = RENDER_CHOICES[key]
            ch, i = imgui.combo(label, ch_list.index(val) if val in ch_list else 0, ch_list)
            out = (ch, ch_list[i])
        elif isinstance(RENDER_DEFAULTS[key], bool):
            out = imgui.checkbox(label, bool(val))
        elif key == 'background':
            out = imgui.color_edit3(label, list(val))
        elif key == 'max_dots':
            ch, v = imgui.slider_int(label, int(val), 0, 5000)
            out = (ch, int(v))
        else:
            lo, hi, lg = RENDER_RANGES.get(key, (0.0, 1.0, False))
            flags = imgui.SliderFlags_.logarithmic.value if lg else 0
            out = imgui.slider_float(label, float(val), lo, hi, '%.4g', flags)
        tip(RENDER_TIPS.get(key, ''))
        return out

    # ------------------------------------------------------------------ render settings
    def gui_render(self):
        r = self.project.render
        imgui.set_next_item_width(-110)
        if imgui.begin_combo('Preset##render', 'choose...'):
            for name in RENDER_PRESETS:
                if imgui.selectable(name, False)[0]:
                    bg = r['background']
                    r.update(dict(RENDER_PRESETS[name]))
                    r['background'] = bg
                    self.touch()
            imgui.end_combo()
        tip('Classic v5 = the original look. Punchy beats = running loudness + dB so hits stand out. '
            'Smooth bass = long smoothing and glow.')
        imgui.push_item_width(-150)
        for title, keys in RENDER_GROUPS:
            if imgui.collapsing_header(title, imgui.TreeNodeFlags_.default_open.value if title in
                                       ('Amplitude', 'Dots & look', 'Time') else 0):
                for key in keys:
                    ch, v = self.render_widget(key, r[key])
                    if ch:
                        r[key] = v
                        self.touch()
        imgui.pop_item_width()
        if r['freq_max'] <= r['freq_min'] * 1.5:
            r['freq_max'] = r['freq_min'] * 1.5

    # ------------------------------------------------------------------ analysis
    def gui_analysis(self):
        a = self.project.analysis
        applied = sanitize_settings(self.manager.settings)
        pending = sanitize_settings(a) != applied
        imgui.set_next_item_width(-110)
        if imgui.begin_combo('Preset##analysis', 'choose...'):
            for name, s in ANALYSIS_PRESETS.items():
                if imgui.selectable(name, False)[0]:
                    a.clear()
                    a.update(s)
                    self.touch()
            imgui.end_combo()
        imgui.push_item_width(-150)
        changed = False
        ws = [str(w) for w in WINDOW_SIZES]

        def wcombo(label, key):
            nonlocal changed
            cur = ws.index(str(a[key])) if str(a[key]) in ws else 2
            ch, i = imgui.combo(label, cur, ws)
            tip(ANALYSIS_TIPS[key])
            if ch:
                a[key] = WINDOW_SIZES[i]
                changed = True
        wcombo('Window size', 'window')
        overlaps = [2, 4, 8, 16]
        cur_ov = max(1, round(a['window'] / max(a['hop'], 1)))
        ov_labels = [f'{o}x (hop {a["window"] // o})' for o in overlaps]
        ch, i = imgui.combo('Overlap', overlaps.index(cur_ov) if cur_ov in overlaps else 1, ov_labels)
        tip(ANALYSIS_TIPS['hop'])
        if ch:
            a['hop'] = a['window'] // overlaps[i]
            changed = True
        ch, i = imgui.combo('Window function', WINDOW_FUNCS.index(a['window_fn']), WINDOW_FUNCS)
        tip(ANALYSIS_TIPS['window_fn'])
        if ch:
            a['window_fn'] = WINDOW_FUNCS[i]
            changed = True
        pads = [1, 2, 4, 8]
        ch, i = imgui.combo('Zero padding', pads.index(a['pad']) if a['pad'] in pads else 0, [f'{p}x' for p in pads])
        tip(ANALYSIS_TIPS['pad'])
        if ch:
            a['pad'] = pads[i]
            changed = True
        ch, v = imgui.checkbox('Multi-resolution', a['multires'])
        tip(ANALYSIS_TIPS['multires'])
        if ch:
            a['multires'] = v
            changed = True
        if a['multires']:
            wcombo('Bass window', 'bass_window')
            wcombo('Treble window', 'treble_window')
            for key, lo, hi in (('cross_low', 40.0, 2000.0), ('cross_high', 500.0, 12000.0)):
                ch, v = imgui.slider_float(key.replace('_', ' ') + ' Hz', float(a[key]), lo, hi, '%.0f',
                                           imgui.SliderFlags_.logarithmic.value)
                tip(ANALYSIS_TIPS[key])
                if ch:
                    a[key] = v
                    changed = True
            ch, v = imgui.slider_float('Blend (octaves)', float(a['blend_oct']), 0.05, 2.0)
            tip(ANALYSIS_TIPS['blend_oct'])
            if ch:
                a['blend_oct'] = v
                changed = True
        imgui.pop_item_width()
        if changed:
            self.touch()
        s = sanitize_settings(a)
        hop_ms = s['hop'] / A.SR * 1000
        imgui.text_disabled(f'{A.SR / s["hop"]:.0f} analysis frames/s ({hop_ms:.1f} ms), '
                            f'{A.SR / s["window"]:.1f} Hz bins')
        imgui.begin_disabled(not pending)
        if imgui.button('Apply (re-analyse spectrum)'):
            self.manager.set_settings(a)
            self.log('Re-analysing spectrum with new settings...')
        imgui.end_disabled()
        if pending:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(1, 0.8, 0.3, 1), 'changes not applied')
        imgui.separator_text('Frame alignment (preview + export)')
        from ..project import ALIGN_LABELS, ALIGN_MODES
        al = self.project.align
        imgui.push_item_width(-150)
        ch, i = imgui.combo('Alignment', ALIGN_MODES.index(al.get('mode', 'centred')), ALIGN_LABELS)
        if ch:
            al['mode'] = ALIGN_MODES[i]
            self.touch()
        tip('Which audio a frame shows. Centred: the frame at time t analyses a window centred on t (t - N/2 .. '
            't + N/2), so a drum hit peaks exactly when it happens.\nLive-style: the window ENDS at t, like the '
            'real-time v5 (a frame appeared after its whole chunk was captured), so dots lag by half a window.\n'
            'Custom: window centre at t + offset. Multi-resolution bands are each aligned for their own size.\n'
            'This is part of the analysis and applies to export too (unlike the preview latency).')
        imgui.begin_disabled(al.get('mode') != 'custom')
        ch, v = imgui.slider_float('Analysis offset (ms)', float(al.get('offset_ms', 0.0)), -100.0, 100.0, '%.1f')
        if ch:
            al['offset_ms'] = float(v)
            self.touch()
        tip('Custom alignment: shifts the analysis window centre to t + offset. Positive = dots show audio from '
            'slightly later (they appear earlier). Sub-frame values are interpolated.')
        imgui.end_disabled()
        imgui.pop_item_width()
        imgui.separator_text('Vocals, stems & vowels')
        ch, v = imgui.checkbox('Use Demucs (GPU stem separation)', self.project.use_demucs)
        tip('Separates vocals / drums / bass / other with the htdemucs model on the GPU. Powers the Instrument '
            'Stems mode and gives the vowel tracker a clean vocal.')
        if ch:
            self.project.use_demucs = v
            self.touch()
        srcs = ['auto', 'demucs', 'centre']
        ch, i = imgui.combo('Vocal source', srcs.index(self.project.vocal_source), srcs)
        tip('auto: Demucs vocals, or the centre-channel estimate when Demucs finds no real voice.\n'
            'centre: centre-panned harmonic content (no Demucs).')
        if ch:
            self.project.vocal_source = srcs[i]
            self.touch()
        if (self.manager.use_demucs, self.manager.vocal_source) != (self.project.use_demucs, self.project.vocal_source):
            if imgui.button('Re-run vocal analysis'):
                self.manager.use_demucs = self.project.use_demucs
                self.manager.vocal_source = self.project.vocal_source
                self.manager.redo_vocals()
        imgui.separator_text('Tracks')
        for ta in self.engine.tracks():
            frac = 1.0 if ta.status == 'ready' else (ta.progress if ta.status == 'spectrum' else 0.5)
            imgui.progress_bar(frac, V2(-1, 0), f'{ta.name}: {ta.status}')
            if ta.error:
                imgui.text_colored(imgui.ImVec4(1, 0.4, 0.4, 1), ta.error)
            elif ta.info is not None:
                imgui.text_disabled(f'  {ta.info.decoder}, {ta.info.source_rate} Hz, {ta.info.source_channels} ch, '
                                    f'{ta.info.subtype}, vocals: {ta.vocal_source or "-"}')
        imgui.separator_text('Live analysis')
        self._live_readouts()

    def _live_readouts(self):
        ti, lt = self.engine.locate(self.view_time())
        if ti is None:
            imgui.text_disabled('(playhead outside any track)')
            return
        ta = self.engine.tracks()[ti]
        if ta.feats is None:
            imgui.text_disabled('features pending...')
            return
        fi = ta.feat_index(lt)
        f = ta.feats
        a = max(0, fi - int(A.FEAT_FPS))
        key = int(np.argmax(f['chroma'][a:fi + 1].mean(axis=0)))
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        imgui.text(f'Tempo {float(f["tempo"][0]):.1f} BPM   pitch centre {names[key]}   '
                   f'f0 {float(f["f0"][fi]):.0f} Hz   centroid {float(f["centroid"][fi]):.0f} Hz')
        imgui.progress_bar(float(np.clip(f['onset'][fi], 0, 1)), V2(-1, 0), 'onset')
        if ta.stems is not None:
            row = np.asarray(ta.stems[min(fi, ta.stems.shape[0] - 1)], np.float32)[2:370].mean(axis=0) / 255
            for i, nm in enumerate(A.STEM_ORDER):
                imgui.progress_bar(float(row[i]), V2(-1, 0), nm)
        if ta.vowel is not None:
            v = ta.vowel
            i = min(fi, len(v['voiced']) - 1)
            w = v['weights'][i]
            imgui.text(f'Vowel  voiced {float(v["voiced"][i]):.2f}  openness {float(v["openness"][i]):.2f}  '
                       f'singer scale {float(v["alpha"][0]):.2f}')
            for j, nm in enumerate(VOWELS):
                c = DEFAULT_VOWEL_COLORS[nm]
                imgui.push_style_color(imgui.Col_.plot_histogram.value, imgui.ImVec4(c[0], c[1], c[2], 1))
                imgui.progress_bar(float(w[j]), V2(-1, 0), nm)
                imgui.pop_style_color()
            self._vowel_chart(v, i)

    def _vowel_chart(self, v, i):
        size = min(imgui.get_content_region_avail().x, 260.0)
        p0 = imgui.get_cursor_screen_pos()
        imgui.dummy(V2(size, size * 0.75))
        dl = imgui.get_window_draw_list()
        w, h = size, size * 0.75
        dl.add_rect_filled(V2(p0.x, p0.y), V2(p0.x + w, p0.y + h), col32(0.07, 0.07, 0.09))

        def pos(f1, f2):
            x = (np.log(2600) - np.log(np.clip(f2, 600, 2600))) / (np.log(2600) - np.log(600))
            y = (np.log(np.clip(f1, 220, 950)) - np.log(220)) / (np.log(950) - np.log(220))
            return V2(p0.x + 10 + x * (w - 20), p0.y + 10 + y * (h - 20))
        for j, nm in enumerate(VOWELS):
            c = DEFAULT_VOWEL_COLORS[nm]
            pp = pos(REF_FORMANTS[j, 0], REF_FORMANTS[j, 1])
            dl.add_circle_filled(pp, 9, col32(*c, 0.35))
            dl.add_text(V2(pp.x - 4, pp.y - 7), col32(1, 1, 1, 0.9), nm)
        a = max(0, i - 40)
        for j in range(a, i + 1):
            if v['voiced'][j] > 0.5 and np.isfinite(v['f1'][j]):
                pp = pos(v['f1'][j], v['f2'][j])
                dl.add_circle_filled(pp, 2, col32(0.8, 0.8, 0.8, 0.15 + 0.6 * (j - a) / 41))
        wv = v['weights'][i]
        f1 = float(wv @ REF_FORMANTS[:, 0])
        f2 = float(wv @ REF_FORMANTS[:, 1])
        c = np.array([DEFAULT_VOWEL_COLORS[n] for n in VOWELS]).T @ wv
        dl.add_circle_filled(pos(f1, f2), 6, col32(*c, 0.95))
        dl.add_text(V2(p0.x + 4, p0.y + h - 16), col32(0.6, 0.6, 0.6), 'F2 <-   vowel chart   F1 v')

    # ------------------------------------------------------------------ project
    def gui_project(self):
        proj = self.project
        imgui.text_disabled(proj.path or '(unsaved project)')
        if imgui.button('New'):
            self.new_project()
        imgui.same_line()
        if imgui.button('Open...'):
            self.open_dialog('open_project')
        imgui.same_line()
        if imgui.button('Save'):
            self.save_project()
        imgui.same_line()
        if imgui.button('Save as...'):
            self.open_dialog('save_project')
        imgui.separator_text('Tracks (played in order)')
        if imgui.button('Add audio...  (Ctrl+I)'):
            self.open_dialog('add_audio')
        tip('FLAC (any bit depth / sample rate / channel count), WAV, MP3, OGG, M4A, ... Several files = an album.')
        remove = None
        for i, ta in enumerate(self.engine.tracks()):
            imgui.push_id(f'trk{i}')
            imgui.text(f'{i + 1}. {ta.name}')
            imgui.same_line()
            imgui.text_disabled(f'{fmt_time(ta.duration, False)}  {ta.status}')
            imgui.same_line()
            if imgui.small_button('^'):
                self.move_track(i, -1)
            imgui.same_line()
            if imgui.small_button('v'):
                self.move_track(i, 1)
            imgui.same_line()
            if imgui.small_button('remove'):
                remove = i
            imgui.pop_id()
        if remove is not None:
            self.remove_track(remove)
        imgui.set_next_item_width(-150)
        ch, v = imgui.slider_float('Gap between tracks (s)', proj.gap_s, 0.0, 10.0)
        if ch:
            proj.gap_s = v
            self.touch()
        imgui.text_disabled(f'Total length {fmt_time(self.total())}')
        imgui.separator_text('Export video')
        ex = proj.export
        imgui.push_item_width(-150)
        names = list(RESOLUTIONS)
        cur = next((n for n, r in RESOLUTIONS.items() if r == (ex['width'], ex['height'])), None)
        if imgui.begin_combo('Resolution', cur or f'{ex["width"]}x{ex["height"]} (custom)'):
            for n in names:
                if imgui.selectable(n, n == cur)[0]:
                    ex['width'], ex['height'] = RESOLUTIONS[n]
                    self.touch()
            imgui.end_combo()
        ch, v = imgui.input_int('Width', int(ex['width']), 16, 64)
        if ch:
            ex['width'] = int(np.clip(v, 64, 7680))
            self.touch()
        ch, v = imgui.input_int('Height', int(ex['height']), 16, 64)
        if ch:
            ex['height'] = int(np.clip(v, 64, 4320))
            self.touch()
        fl = [str(f) for f in FPS_CHOICES]
        ch, i = imgui.combo('Frame rate', fl.index(str(ex['fps'])) if str(ex['fps']) in fl else 2, fl)
        if ch:
            ex['fps'] = FPS_CHOICES[i]
            self.touch()
        tip('Rendered offline frame by frame: frame k shows t = k / fps exactly.')
        ch, i = imgui.combo('Video codec', CODECS.index(ex['codec']) if ex['codec'] in CODECS else 0, CODECS)
        if ch:
            ex['codec'] = CODECS[i]
            self.touch()
        tip('auto = NVENC if the driver supports it, otherwise libx264.')
        alabels = ['Copy original (bit-exact, same size)', 'AAC 320k', 'FLAC (lossless)',
                   'PCM 24-bit (lossless, .mov/.mkv)']
        ch, i = imgui.combo('Audio', AUDIO_CODECS.index(ex['audio']) if ex['audio'] in AUDIO_CODECS else 0, alabels)
        if ch:
            ex['audio'] = AUDIO_CODECS[i]
            self.touch()
        tip("Copy original puts the source file's audio stream into the video untouched: no quality loss and no\n"
            'size increase. Needs a single track; otherwise lossless sources become FLAC and lossy ones AAC.')
        ch, v = imgui.slider_int('Quality (CRF/CQ)', int(ex['quality']), 10, 35)
        if ch:
            ex['quality'] = v
            self.touch()
        tip('Lower = better quality, bigger file. 18 is visually lossless for most content.')
        al = self.project.align
        imgui.text_disabled(f'Frame alignment: {al["mode"]}'
                            + (f' {al["offset_ms"]:+.0f} ms' if al['mode'] == 'custom' else '')
                            + ' (Analysis panel). Preview latency is not applied.')
        ch, v = imgui.input_text('Output file', ex.get('path', ''))
        if ch:
            ex['path'] = v
            self.touch()
        imgui.same_line()
        if imgui.small_button('...'):
            self.open_dialog('export_path')
        imgui.pop_item_width()
        imgui.begin_disabled(self.total() <= 0 or self.exporter is not None and not self.exporter.done)
        if imgui.button('Export whole timeline  (Ctrl+E)', V2(-1, 0)):
            self.start_export()
        imgui.end_disabled()
        if self.export_msg:
            imgui.push_text_wrap_pos(0)
            imgui.text_colored(imgui.ImVec4(1, 0.6, 0.4, 1), self.export_msg)
            imgui.pop_text_wrap_pos()

    def export_modal(self):
        ex = self.exporter
        if ex is None:
            return
        if not imgui.is_popup_open('Exporting'):
            imgui.open_popup('Exporting')
        vp = imgui.get_main_viewport()
        imgui.set_next_window_pos(V2(vp.work_pos.x + vp.work_size.x / 2, vp.work_pos.y + vp.work_size.y / 2),
                                  imgui.Cond_.always.value, V2(0.5, 0.5))
        imgui.set_next_window_size(V2(520, 0))
        opened, _ = imgui.begin_popup_modal('Exporting', None, imgui.WindowFlags_.always_auto_resize.value)
        if opened:
            imgui.text(os.path.basename(ex.path))
            imgui.text_disabled(f'{ex.w}x{ex.h} @ {ex.fps} fps, {ex.n_frames} frames, '
                                f'{getattr(ex, "codec_used", "?")} + {getattr(ex, "audio_used", ex.audio)}')
            if getattr(ex, 'audio_note', ''):
                imgui.text_disabled('Audio: ' + ex.audio_note)
            eta = ex.eta()
            imgui.progress_bar(ex.progress, V2(-1, 0), f'{ex.k} / {ex.n_frames}' + (f'   ETA {eta:.0f}s' if eta else ''))
            if ex.done or ex.error or ex.cancelled:
                if ex.error:
                    imgui.push_text_wrap_pos(500)
                    imgui.text_colored(imgui.ImVec4(1, 0.4, 0.4, 1), ex.error)
                    imgui.pop_text_wrap_pos()
                elif ex.done:
                    imgui.text_colored(imgui.ImVec4(0.5, 1, 0.5, 1), 'Done.')
                    if imgui.button('Open folder'):
                        os.startfile(os.path.dirname(os.path.abspath(ex.path)))
                    imgui.same_line()
                if imgui.button('Close'):
                    self.export_msg = ex.error or ('Exported ' + ex.path if ex.done else 'Export cancelled')
                    self.exporter = None
                    imgui.close_current_popup()
            else:
                if imgui.button('Cancel'):
                    ex.cancel()
            imgui.end_popup()

    # ------------------------------------------------------------------ menus / fullscreen
    def menus(self):
        if imgui.begin_menu('File'):
            if imgui.menu_item('New project', 'Ctrl+N', False)[0]:
                self.new_project()
            if imgui.menu_item('Open project...', 'Ctrl+O', False)[0]:
                self.open_dialog('open_project')
            if imgui.menu_item('Save project', 'Ctrl+S', False)[0]:
                self.save_project()
            if imgui.menu_item('Save project as...', 'Ctrl+Shift+S', False)[0]:
                self.open_dialog('save_project')
            imgui.separator()
            if imgui.menu_item('Add audio...', 'Ctrl+I', False)[0]:
                self.open_dialog('add_audio')
            if imgui.menu_item('Export video...', 'Ctrl+E', False)[0]:
                self.start_export()
            imgui.end_menu()
        if imgui.begin_menu('Edit'):
            if imgui.menu_item('Undo', 'Ctrl+Z', False, bool(self.history.undo))[0]:
                self.undo()
            if imgui.menu_item('Redo', 'Ctrl+Y', False, bool(self.history.redo))[0]:
                self.redo()
            imgui.separator()
            if imgui.menu_item('Add keyframe at playhead', 'K', False)[0]:
                self.add_keyframe(self.playhead)
            if imgui.menu_item('Duplicate keyframe at playhead', 'Ctrl+D', False)[0]:
                self.duplicate_keyframe(self.selected_kf, self.playhead)
            if imgui.menu_item('Delete keyframe', 'Del', False, len(self.project.keyframes) > 1)[0]:
                self.delete_keyframe(self.selected_kf)
            imgui.end_menu()
        if imgui.begin_menu('Playback'):
            if imgui.menu_item('Play / pause', 'Space', self.player.playing)[0]:
                self.toggle_play()
            if imgui.menu_item('Fullscreen preview', 'F', self.fullscreen)[0]:
                self.fullscreen = not self.fullscreen
            if imgui.menu_item('Mute', 'M', self.player.muted)[0]:
                self.player.muted = not self.player.muted
            imgui.end_menu()
        if imgui.begin_menu('Help'):
            if imgui.menu_item('Keyboard shortcuts', 'F1', self.show_help)[0]:
                self.show_help = not self.show_help
            imgui.end_menu()

    def show_gui(self):
        self.shortcuts()
        self.export_modal()
        if self.fullscreen:
            vp = imgui.get_main_viewport()
            imgui.set_next_window_pos(vp.pos)
            imgui.set_next_window_size(vp.size)
            flags = (imgui.WindowFlags_.no_decoration.value | imgui.WindowFlags_.no_saved_settings.value |
                     imgui.WindowFlags_.no_docking.value | imgui.WindowFlags_.no_move.value)
            imgui.push_style_var(imgui.StyleVar_.window_padding.value, V2(0, 0))
            imgui.begin('##fullscreen', None, flags)
            imgui.set_window_focus()
            av = imgui.get_content_region_avail()
            self._draw_view(av.x, av.y, hud=False)
            imgui.end()
            imgui.pop_style_var()
        if imgui.is_key_pressed(imgui.Key.f1, False):
            self.show_help = not self.show_help
        if self.show_help:
            imgui.set_next_window_size(V2(460, 0), imgui.Cond_.first_use_ever.value)
            opened, self.show_help = imgui.begin('Keyboard shortcuts', self.show_help)
            if opened:
                for k, d in SHORTCUTS:
                    imgui.text(f'{k:<18}')
                    imgui.same_line(170)
                    imgui.text_disabled(d)
            imgui.end()


SHORTCUTS = [
    ('Space', 'Play / pause'), ('Left / Right', 'Step 1/30 s (Shift: 1 s)'), ('Ctrl+Left / Right', 'Previous / next keyframe'),
    ('Home / End', 'Start / end'), ('K', 'Add keyframe at playhead'), ('Ctrl+D', 'Duplicate keyframe at playhead'),
    ('Delete', 'Delete selected keyframe'), ('Ctrl+Z / Ctrl+Y', 'Undo / redo'), ('Ctrl+S', 'Save (Shift: save as)'),
    ('Ctrl+O / Ctrl+N', 'Open / new project'), ('Ctrl+I', 'Add audio'), ('Ctrl+E', 'Export video'),
    ('F', 'Fullscreen preview (Esc to leave)'), ('M', 'Mute'), ('F1', 'This help'),
    ('Timeline', 'drag ruler = scrub, wheel = scroll, Ctrl+wheel = zoom'),
    ('Keyframe lane', 'double-click = add, drag = move (Shift snaps to beats), right-click = menu'),
    ('Palette bar', 'double-click = add stop, drag stop = move, right-click stop = delete'),
]


def build_runner_params(app, headless=False):
    rp = hello_imgui.RunnerParams()
    rp.app_window_params.window_title = 'FFT Dot Studio'
    rp.imgui_window_params.show_menu_bar = True
    rp.imgui_window_params.show_status_bar = False
    rp.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    rp.imgui_window_params.enable_viewports = False
    rp.ini_filename = 'fft-dot-studio.ini'
    rp.callbacks.post_init = app.post_init
    rp.callbacks.before_exit = app.before_exit
    rp.callbacks.pre_new_frame = app.pre_frame
    rp.callbacks.show_menus = app.menus
    rp.callbacks.show_gui = app.show_gui
    rp.fps_idling.fps_idle = 10.0

    def split(initial, new, direction, ratio):
        s = hello_imgui.DockingSplit()
        s.initial_dock, s.new_dock, s.direction, s.ratio = initial, new, direction, ratio
        return s
    rp.docking_params.docking_splits = [
        split('MainDockSpace', 'TimelineSpace', imgui.Dir.down, 0.30),
        split('MainDockSpace', 'LeftSpace', imgui.Dir.left, 0.22),
        split('MainDockSpace', 'RightSpace', imgui.Dir.right, 0.30),
    ]

    def win(label, space, fn):
        w = hello_imgui.DockableWindow()
        w.label, w.dock_space_name, w.gui_function = label, space, fn
        return w
    rp.docking_params.dockable_windows = [
        win('Viewer', 'MainDockSpace', app.gui_viewer),
        win('Timeline', 'TimelineSpace', app.gui_timeline),
        win('Project', 'LeftSpace', app.gui_project),
        win('Analysis', 'LeftSpace', app.gui_analysis),
        win('Keyframe', 'RightSpace', app.gui_keyframe),
        win('Render', 'RightSpace', app.gui_render),
    ]
    if headless:
        rp.platform_backend_type = hello_imgui.PlatformBackendType.null
        rp.renderer_backend_type = hello_imgui.RendererBackendType.null
        rp.ini_disable = True
    else:
        from ..monitor import set_dpi_aware, smallest_monitor
        set_dpi_aware()
        mon = smallest_monitor()
        geo = rp.app_window_params.window_geometry
        if mon:
            x, y, w, h = mon['work']
            geo.window_size_measure_mode = hello_imgui.WindowSizeMeasureMode.screen_coords
            geo.position_mode = hello_imgui.WindowPositionMode.from_coords
            geo.position = (x + 20, y + 20)
            geo.size = (max(800, w - 40), max(600, h - 40))
            geo.window_size_state = hello_imgui.WindowSizeState.maximized
        else:
            geo.size = (1600, 960)
    return rp


def run(project_path=None, tracks=None):
    app = StudioApp(project_path, tracks)
    hello_imgui.run(build_runner_params(app))
