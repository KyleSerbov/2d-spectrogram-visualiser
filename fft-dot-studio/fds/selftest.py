"""Headless self-test: never opens a window, never plays audio.

Covers colour maths, FLAC/WAV import, analysis + beats, the vowel tracker on synthetic vowels, every colour
mode (native + palette), tweening/morphing, determinism, render/analysis presets, save/load, frame-exact
export checked with ffprobe, and a scripted run of the whole editor UI on null ImGui backends."""
import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}  {detail}', flush=True)
    return ok


def info(name, detail=''):
    print(f'  [INFO] {name}  {detail}', flush=True)


def section(title):
    print(f'\n== {title}', flush=True)


def run(out_dir=None, quick=False, gui=True):
    out_dir = out_dir or os.path.join(tempfile.gettempdir(), 'fds_selftest')
    os.makedirs(out_dir, exist_ok=True)
    os.environ.setdefault('FDS_CACHE', os.path.join(out_dir, 'cache'))
    import moderngl
    import soundfile as sf
    from PIL import Image

    from . import analysis as A
    from . import synth
    from .colorspace import mix_oklab, oklab_to_srgb, srgb_to_oklab
    from .engine import Engine
    from .export import Exporter, probe
    from .modes import MODE_ORDER, MODES
    from .palettes import PRESETS, Palette
    from .project import RENDER_PRESETS, Keyframe, Project
    from .renderer import Renderer
    from .vowel import VOWELS, synth_vowel, track_vowels

    t_start = time.time()
    print(f'Self-test output: {out_dir}')
    gl = moderngl.create_standalone_context()

    section('Colour maths and palettes')
    rng = np.random.default_rng(0)
    c = rng.random((500, 3)).astype(np.float32)
    err = np.abs(oklab_to_srgb(srgb_to_oklab(c)) - c).max()
    check('OKLab round trip', err < 2e-3, f'max err {err:.2e}')
    m = mix_oklab(np.array([1, 0, 0], np.float32), np.array([0, 0, 1], np.float32), 0.5)
    check('OKLab mix red/blue is purple-ish', m[0] > 0.3 and m[2] > 0.3 and m[1] < 0.4, str(np.round(m, 3)))
    for name in PRESETS:
        lut = Palette.preset(name).lut()
        if not (lut.shape == (512, 3) and np.isfinite(lut).all()):
            check(f'palette {name}', False)
    check('all palette presets build', True, f'{len(PRESETS)} presets')
    pal = Palette.preset('Rainbow (HSV)')
    check('cyclic palette wraps', np.allclose(pal.sample(np.array([0.0])), pal.sample(np.array([1.0])), atol=1e-6))

    section('Audio import (WAV + 24-bit/96 kHz 6-channel FLAC)')
    song, gt = synth.song(12.0, bpm=120.0)
    wav = os.path.join(out_dir, 'test_song.wav')
    sf.write(wav, song, 44100, subtype='PCM_16')
    from scipy.signal import resample_poly
    x96 = resample_poly(song, 320, 147, axis=0)
    six = np.zeros((len(x96), 6), np.float32)
    six[:, 0], six[:, 1] = x96[:, 0], x96[:, 1]
    six[:, 2] = 0.5 * (x96[:, 0] + x96[:, 1]) * 0.3
    flac = os.path.join(out_dir, 'test_song_24bit_96k_6ch.flac')
    sf.write(flac, six, 96000, subtype='PCM_24', format='FLAC')
    finfo = sf.info(flac)
    check("wrote 24-bit 96 kHz 6ch FLAC", finfo.subtype == "PCM_24" and finfo.samplerate == 96000 and finfo.channels == 6,
          f"{finfo.subtype} {finfo.samplerate} Hz {finfo.channels} ch")
    mgr = A.AnalysisManager(sync=True)
    t0 = time.time()
    ta_w = mgr.get(wav)
    check('WAV analysed', ta_w.status == 'ready' and ta_w.error is None, f'{time.time() - t0:.1f}s ({A.DEVICE})')
    t0 = time.time()
    ta_f = mgr.get(flac)
    ok = (ta_f.status == 'ready' and ta_f.error is None and ta_f.info.decoder == 'libsndfile'
          and ta_f.info.source_rate == 96000 and ta_f.info.source_channels == 6 and abs(ta_f.duration - 12.0) < 0.01)
    check('FLAC imported via libsndfile, resampled + downmixed, analysed', ok,
          f'{ta_f.info.decoder} {ta_f.info.source_rate}->{A.SR} Hz, {ta_f.info.source_channels}->2 ch, '
          f'{ta_f.duration:.3f}s, {time.time() - t0:.1f}s')
    a = ta_w.float_audio()
    b = ta_f.float_audio()
    n = min(len(a), len(b))
    corr = np.corrcoef(a[:n, 0], b[:n, 0])[0, 1]
    check('FLAC downmix matches source', corr > 0.98, f'corr {corr:.4f}')
    mp3 = os.path.join(out_dir, 'test_song.mp3')
    try:
        import subprocess
        from .audio_io import find_ffmpeg, load_audio
        subprocess.run([find_ffmpeg(), '-y', '-v', 'error', '-i', wav, '-b:a', '192k', mp3], check=True,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        xm, im = load_audio(mp3)
        check('MP3 import', abs(im.duration - 12.0) < 0.1, f'{im.decoder}, {im.duration:.3f}s')
        m4a = os.path.join(out_dir, 'test_song.m4a')
        subprocess.run([find_ffmpeg(), '-y', '-v', 'error', '-i', wav, '-c:a', 'aac', m4a], check=True,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        xa, ia = load_audio(m4a)
        check('M4A/AAC import (ffmpeg fallback)', ia.decoder == 'ffmpeg' and abs(ia.duration - 12.0) < 0.1,
              f'{ia.decoder}, {ia.duration:.3f}s')
    except Exception as e:
        check('MP3/M4A import', False, str(e))

    section('Features and beat tracking')
    beats = ta_w.feats['beats']
    errs = np.array([np.min(np.abs(gt['beats'] - bt)) for bt in beats]) if len(beats) else np.array([1.0])
    check('tempo ~120 BPM', abs(float(ta_w.feats['tempo'][0]) - 120) < 3, f'{float(ta_w.feats["tempo"][0]):.2f} BPM')
    check('beats within 20 ms of ground truth', np.median(errs) < 0.02 and len(beats) >= 0.8 * len(gt['beats']),
          f'{len(beats)} beats, median err {np.median(errs) * 1000:.1f} ms')
    check('FLAC tempo ~120 BPM', abs(float(ta_f.feats['tempo'][0]) - 120) < 3, f'{float(ta_f.feats["tempo"][0]):.2f} BPM')
    check('Demucs stems computed', ta_w.stems is not None, f'source for vowels: {ta_w.vocal_source}')

    section('Vowel tracker (synthetic sung vowels, perturbed singers)')
    accs = []
    for scale in (0.92, 1.0, 1.12):
        for f0 in ((110, 220) if quick else (110, 165, 220, 262, 330)):
            sig, lab = [], []
            for i, v in enumerate(VOWELS * 2):
                s = synth_vowel(v, f0=f0, dur=0.6, vibrato=0.015, seed=i + f0, formant_scale=scale, jitter=0.06)
                sig.append(s)
                lab += [v] * len(s)
            x = np.concatenate(sig)
            lab = np.array(lab)
            times = np.arange(0, len(x) / 44100, 512 / 44100)
            r = track_vowels(x, 44100, times)
            truth = lab[np.minimum((times * 44100).astype(int), len(lab) - 1)]
            pred = np.array(VOWELS)[r['weights'].argmax(1)]
            seg = (times % 0.6 > 0.08) & (times % 0.6 < 0.52)
            mm = seg & (r['voiced'] > 0.5)
            acc = float((pred[mm] == truth[mm]).mean()) if mm.any() else 0.0
            accs.append(acc)
            print(f'     singer scale {scale:.2f}  f0 {f0:3d} Hz  acc {acc:.3f}  (scale est {r["alpha"][0]:.2f})')
    check('vowel accuracy >= 90% on average', np.mean(accs) >= 0.9, f'mean {np.mean(accs):.3f}')
    vx = ta_w.vowel
    vt = []
    for (s0, s1, v) in gt['vowels']:
        fr = np.arange(int((s0 + 0.1) * A.FEAT_FPS), int((s1 - 0.1) * A.FEAT_FPS))
        vm = vx['voiced'][fr] > 0.5
        vt.append(((np.array(VOWELS)[vx['weights'][fr].argmax(1)][vm] == v).mean()) if vm.any() else 0)
    info('vowels inside a dense synthetic mix',
         f'acc {np.mean(vt):.2f} via {ta_w.vocal_source} (synthetic buzz voices separate poorly; clean voice ~0.8-1.0)')

    section('Frame alignment (click test)')
    _click_test(out_dir, sf, A)

    section('Every colour mode (render to PNG)')
    proj = Project()
    proj.tracks = [wav]
    eng = Engine(proj, mgr)
    ren = Renderer(gl, 640, 360)
    mode_dir = os.path.join(out_dir, 'modes')
    os.makedirs(mode_dir, exist_ok=True)
    tiles = []
    bad = []
    for key in MODE_ORDER:
        for up in ((False, True) if MODES[key].native else (False,)):
            proj.keyframes = [Keyframe(0.0, key)]
            proj.keyframes[0].use_palette = up
            if up:
                proj.keyframes[0].palette = Palette.preset('Neon')
            proj.revision += 1
            try:
                d = eng.gather(3.13, 16 / 9)
                ren.render(d)
                img = ren.read_rgb()
                Image.fromarray(img).save(os.path.join(mode_dir, f'{key}{"_palette" if up else ""}.png'))
                if not up:
                    tiles.append((MODES[key].name, img))
                if len(d['x']) == 0 or img.max() < 30:
                    bad.append(f'{key}{"/pal" if up else ""}: empty')
            except Exception as e:
                traceback.print_exc()
                bad.append(f'{key}: {e}')
    check(f'all {len(MODE_ORDER)} modes render (native + palette variants)', not bad, '; '.join(bad))
    _contact_sheet(tiles, os.path.join(out_dir, 'modes_sheet.png'))

    section('Keyframe tweening')
    kA, kB = Keyframe(2.0, 'note_rainbow'), Keyframe(6.0, 'radial')
    kA.easing = 'linear'
    kA.transition = 'crossfade'
    proj.keyframes = [kA, kB]
    proj.revision += 1
    frames = []
    for u in np.linspace(0, 1, 6):
        t = 2.0 + 4.0 * u
        d = eng.gather(t, 16 / 9)
        frames.append((u, d))
    # colour of the same dots under A only, B only, and the blend
    ti, lt = eng.locate(4.0)
    f = int(lt / (eng.tracks()[ti].spectrum.hop / A.SR))
    ctx = eng.ctx(ti, f, proj.render_at(4.0), 16 / 9)
    layers, (k0, k1, u) = eng.layers_at(4.0)
    rgbA = eng.colourise(ctx, [layers[0]])[0]
    rgbB = eng.colourise(ctx, [(1.0, layers[1][1])])[0]
    rgbM = eng.colourise(ctx, layers)[0]
    vis = ctx.visible
    dA = np.abs(srgb_to_oklab(rgbM[vis]) - srgb_to_oklab(rgbA[vis])).mean()
    dB = np.abs(srgb_to_oklab(rgbM[vis]) - srgb_to_oklab(rgbB[vis])).mean()
    check('crossfade at u=0.5 sits between both modes', abs(u - 0.5) < 1e-6 and dA > 0.005 and dB > 0.005 and abs(dA - dB) < 0.5 * max(dA, dB),
          f'u={u:.2f} dist->A {dA:.3f} dist->B {dB:.3f}')
    strip = np.concatenate([_render(eng, ren, 2.0 + 4.0 * uu) for uu in np.linspace(0, 1, 6)], axis=1)
    Image.fromarray(strip).save(os.path.join(out_dir, 'tween_strip.png'))
    kC, kD = Keyframe(0.0, 'radial'), Keyframe(4.0, 'radial')
    kC.palette, kD.palette = Palette.preset('Fire'), Palette.preset('Ice')
    kC.set_param('radius', 0.2)
    kD.set_param('radius', 1.2)
    proj.keyframes = [kC, kD]
    proj.revision += 1
    layers, _ = eng.layers_at(2.0)
    check('same-mode morph interpolates parameters', len(layers) == 1 and abs(layers[0][1].params['radius'] - 0.7) < 1e-6,
          f'radius at midpoint {layers[0][1].params["radius"]:.3f}')
    mid = layers[0][1].pal.lut.mean(axis=0)
    check('palette morph blends LUTs', np.all(np.abs(mid - (kC.palette.lut().mean(0) + kD.palette.lut().mean(0)) / 2) < 0.15),
          str(np.round(mid, 3)))
    from .project import ease
    check('easing curves', ease(0.5, 'linear') == 0.5 and ease(0.5, 'hold') == 0.0 and ease(0.5, 'smooth') == 0.5
          and ease(0.2, 'ease in') < 0.2 < ease(0.2, 'ease out'))

    section('Determinism (scrub = playback = export)')
    eng2 = Engine(proj, mgr)
    da = eng.gather(5.4321, 16 / 9)
    for t in (9.0, 1.0, 5.0):
        eng.gather(t, 16 / 9)
    db = eng.gather(5.4321, 16 / 9)
    dc = eng2.gather(5.4321, 16 / 9)
    same = all(np.array_equal(da[k], db[k]) and np.array_equal(da[k], dc[k]) for k in ('x', 'y', 'i', 'rgb'))
    check('identical frames regardless of scrub history / cache', same, f'{len(da["x"])} dots')

    section('Render + analysis presets')
    proj.keyframes = [Keyframe(0.0, 'note_rainbow')]
    for name, preset in RENDER_PRESETS.items():
        proj.render = dict(preset)
        proj.revision += 1
        img = _render(eng, ren, 3.5)
        Image.fromarray(img).save(os.path.join(out_dir, f'render_preset_{name.replace(" ", "_")}.png'))
        check(f'render preset "{name}"', img.max() > 30, f'mean {img.mean():.2f}')
    proj.render = dict(RENDER_PRESETS['Classic v5'])
    for key in ('freq_scale', 'pan_mode', 'mirror_x', 'flip_y'):
        pass
    for name, s in A.ANALYSIS_PRESETS.items():
        mgr.set_settings(s)
        proj.revision += 1
        img = _render(eng, ren, 3.5)
        sp = eng.tracks()[0].spectrum
        check(f'analysis preset "{name}"', img.max() > 30, f'{len(sp.bands)} band(s), {len(sp.freqs)} bins/frame, '
              f'hop {sp.hop}')
    mgr.set_settings(A.ANALYSIS_DEFAULTS)
    for key, val in (('freq_scale', 'mel'), ('freq_scale', 'linear'), ('pan_mode', 'phase'), ('mirror_x', True),
                     ('mirror_y', True), ('normalise', 'running'), ('amp_scale', 'db'), ('peak_pick', True)):
        proj.render = dict(RENDER_PRESETS['Classic v5'])
        proj.render[key] = val
        proj.render['max_dots'] = 300 if key == 'peak_pick' else 0
        proj.revision += 1
        d = eng.gather(3.5, 16 / 9)
        check(f'render option {key}={val}', len(d['x']) > 0, f'{len(d["x"])} dots')
    proj.render = dict(RENDER_PRESETS['Classic v5'])
    proj.render['smooth_release_ms'] = 200.0
    proj.revision += 1
    d1 = eng.gather(3.5, 16 / 9)
    check('temporal smoothing', len(d1['x']) > 0)
    proj.render = dict(RENDER_PRESETS['Classic v5'])

    section('Project save / load')
    proj.keyframes = [Keyframe(0.0, 'vowel'), Keyframe(3.0, 'beat_zones'), Keyframe(7.0, 'radial')]
    proj.keyframes[2].palette = Palette.preset('Synthwave')
    proj.keyframes[2].set_param('centre', 'centroid')
    proj.keyframes[1].render_overrides = {'brightness': 20.0, 'background': [0.1, 0.0, 0.2]}
    proj.align = {'mode': 'custom', 'offset_ms': -42.0}
    proj.preview_latency = {'mode': 'manual', 'ms': 35.0}
    proj.export['fps'] = 25
    pth = os.path.join(out_dir, 'roundtrip.fdsproj')
    proj.save(pth)
    p2 = Project.load(pth)
    a1, a2 = proj.to_dict(), p2.to_dict()
    check('save -> load round trip is identical', json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True))
    r = p2.render_at(5.0)
    check('render overrides tween', 8.0 < r['brightness'] < 20.0, f'brightness at t=5: {r["brightness"]:.2f}')

    section('Export only waits for the analyses the colour modes use')
    ta0 = eng.tracks()[0]
    saved = (proj.keyframes, ta0.feats, ta0.status)
    try:
        ta0.feats, ta0.status = None, 'features'
        cases = [([Keyframe(0.0, 'mono')], False), ([Keyframe(0.0, 'radial')], False),
                 ([Keyframe(0.0, 'onset_flash')], True), ([Keyframe(0.0, 'vowel')], True)]
        pulse = Keyframe(0.0, 'radial')
        pulse.set_param('pulse', 0.5)
        cases.append(([pulse], True))
        results = []
        for kfs, should_block in cases:
            proj.keyframes = kfs
            b = eng.export_blockers()
            results.append((kfs[0].mode, bool(b) == should_block, b))
        check('mono / radial export while features + vowels are still running',
              all(ok for _m, ok, _b in results), '; '.join(f'{m}: {"blocked" if b else "free"}' for m, _ok, b in results))
        proj.keyframes = [Keyframe(0.0, 'mono')]
        dots = eng.gather(5.0, 16 / 9)
        check('rendering works without features', len(dots['x']) > 0, f'{len(dots["x"])} dots')
    finally:
        proj.keyframes, ta0.feats, ta0.status = saved

    section('Export (offline, frame exact, ffprobe)')
    from .audio_io import find_ffmpeg
    from .export import nvenc_works
    check('GPU encoder (NVENC) available', nvenc_works('h264_nvenc'), find_ffmpeg())
    proj.keyframes = [Keyframe(0.0, 'note_rainbow'), Keyframe(3.0, 'radial')]
    proj.revision += 1
    ex = Exporter(eng, gl, os.path.join(out_dir, 'export_nonblocking.mp4'), 640, 360, 30, 'auto', 'aac', 23)
    ex.start()
    worst = 0.0
    while True:
        t0 = time.perf_counter()
        running = ex.step(0.03)
        worst = max(worst, time.perf_counter() - t0)
        if not running:
            break
        time.sleep(0.001)
    check('export step never blocks the UI', worst < 0.25 and ex.done and not ex.error,
          f'longest step {worst * 1000:.0f} ms, {ex.codec_used}, {ex.error or "ok"}')
    check('auto export uses the GPU encoder', ex.codec_used == 'h264_nvenc', ex.codec_used)
    for fps, audio, ext in ((30, 'aac', '.mp4'), (25, 'flac', '.mkv'), (24, 'pcm', '.mov')):
        outp = os.path.join(out_dir, f'export_{fps}fps{ext}')
        ex = Exporter(eng, gl, outp, 640, 360, fps, 'auto', audio, 23)
        t0 = time.time()
        try:
            ex.run_blocking()
            pr = probe(outp)
            ok = (pr['frames'] == ex.n_frames and abs(pr['vdur'] - ex.exact_duration) < 1e-3
                  and abs(pr['adur'] - ex.exact_duration) < 0.03 and abs(pr['astart']) < 1e-3 and abs(pr['vstart']) < 1e-3)
            check(f'export {fps} fps + {audio}', ok,
                  f'{pr["frames"]}/{ex.n_frames} frames, video {pr["vdur"]:.3f}s, audio {pr["adur"]:.3f}s '
                  f'({pr["acodec"]}), {ex.codec_used}, {time.time() - t0:.1f}s')
        except Exception as e:
            traceback.print_exc()
            check(f'export {fps} fps + {audio}', False, str(e))
    proj2 = Project()
    proj2.tracks = [wav, flac]
    proj2.keyframes = [Keyframe(0.0, 'beat_pulse')]
    eng_album = Engine(proj2, mgr)
    outp = os.path.join(out_dir, 'export_album.mp4')
    ex = Exporter(eng_album, gl, outp, 480, 270, 30, 'auto', 'aac', 28)
    ex.run_blocking()
    pr = probe(outp)
    check('album (2 tracks) exports whole timeline', pr['frames'] == ex.n_frames and abs(pr['adur'] - 24.0) < 0.05,
          f'{pr["frames"]} frames, audio {pr["adur"]:.3f}s')

    ex = Exporter(eng_album, gl, os.path.join(out_dir, 'export_album_copy.mp4'), 320, 180, 30, 'auto', 'copy', 28)
    ex.run_blocking()
    check('copy audio falls back for an album', ex.audio_used == 'flac', ex.audio_note)

    import subprocess
    from .audio_io import find_ffmpeg

    def packet_hashes(path):
        r = subprocess.run([find_ffmpeg(), '-v', 'error', '-i', path, '-map', '0:a:0', '-c', 'copy', '-f', 'framemd5',
                            '-'], capture_output=True, text=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return [ln.rsplit(',', 1)[-1].strip() for ln in r.stdout.splitlines() if ln and not ln.startswith('#')]

    m4a = os.path.join(out_dir, 'test_song.m4a')
    proj3 = Project()
    proj3.tracks = [m4a]
    proj3.keyframes = [Keyframe(0.0, 'note_rainbow')]
    eng3 = Engine(proj3, mgr)
    outp = os.path.join(out_dir, 'export_copy.mp4')
    ex = Exporter(eng3, gl, outp, 320, 180, 30, 'auto', 'copy', 28)
    ex.run_blocking()
    src_h, out_h = packet_hashes(m4a), packet_hashes(outp)
    pr = probe(outp)
    n = len(out_h)
    check('copy audio is bit-identical to the source', ex.audio_used == 'copy' and n > 0 and out_h == src_h[:n]
          and len(src_h) - n <= 2 and pr['frames'] == ex.n_frames,
          f'{ex.audio_note}; {n}/{len(src_h)} packets identical, {pr["frames"]} frames, audio {pr["adur"]:.3f}s')

    if gui:
        section('Editor UI (headless ImGui null backends, scripted input)')
        try:
            gui_smoke(wav, flac, gl, out_dir)
        except Exception as e:
            traceback.print_exc()
            check('GUI smoke test', False, str(e))

    section('Summary')
    fails = [r for r in RESULTS if not r[1]]
    print(f'{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed in {time.time() - t_start:.0f}s')
    for name, _ok, det in fails:
        print(f'  FAILED: {name}  {det}')
    print(f'Images and videos: {out_dir}')
    return 0 if not fails else 1


def _click_test(out_dir, sf, A):
    """A click at time T must peak in the frame displayed at T (centred alignment), at T + N/2 for the
    live-style alignment (per band for multi-resolution), and at T - offset for a custom offset."""
    from .engine import Engine
    from .project import Keyframe, Project
    sr = A.SR
    T = 1.5
    x = np.zeros((int(3 * sr), 2), np.float32)
    i = int(T * sr)
    x[i:i + 20] = 0.9
    path = os.path.join(out_dir, 'click.wav')
    sf.write(path, x, sr, subtype='PCM_24')
    for preset in ('Classic v5', 'Smooth bass'):
        mgr = A.AnalysisManager(use_demucs=False, sync=True)
        mgr.set_settings(A.ANALYSIS_PRESETS[preset])
        ta = mgr.get(path)
        proj = Project()
        proj.tracks = [path]
        proj.keyframes = [Keyframe(0.0, 'mono')]
        eng = Engine(proj, mgr)
        sp = ta.spectrum
        hop_s = sp.hop / sr
        lens = [len(b.freqs) for b in sp.bands]
        edges = np.cumsum([0] + lens)
        frames = range(int((T - 0.4) / hop_s), int((T + 0.4) / hop_s))
        for mode, off in (('centred', 0.0), ('live', 0.0), ('custom', 50.0), ('custom', -30.0)):
            proj.align = {'mode': mode, 'offset_ms': off}
            proj.revision += 1
            E = np.array([[float((fr['L'][edges[b]:edges[b + 1]] + fr['R'][edges[b]:edges[b + 1]]).sum())
                           for b in range(len(lens))] for fr in (eng.raw_frame(ta, f) for f in frames)])
            fs = np.array(list(frames))
            res = []
            ok = True
            for b, band in enumerate(sp.bands):
                peak_t = fs[int(np.argmax(E[:, b]))] * hop_s
                if mode == 'centred':
                    exp = T
                elif mode == 'live':
                    exp = T + band.N / 2 / sr
                else:
                    exp = T - off / 1000.0
                err = peak_t - exp
                ok &= abs(err) <= hop_s / 2 + 1e-6
                res.append(f'N={band.N}: peak {peak_t * 1000:.1f} ms (expect {exp * 1000:.1f})')
            name = {'centred': 'Centred, offset 0', 'live': 'Live-style',
                    'custom': f'Custom {off:+.0f} ms'}[mode]
            check(f'click alignment [{preset}] {name}', ok, '; '.join(res))


def _render(eng, ren, t):
    ren.render(eng.gather(t, ren.size[0] / ren.size[1]))
    return ren.read_rgb().copy()


def _contact_sheet(tiles, path, cols=5):
    from PIL import Image, ImageDraw
    if not tiles:
        return
    rows = (len(tiles) + cols - 1) // cols
    W, H = 320, 180
    sheet = Image.new('RGB', (cols * W, rows * (H + 18)), (30, 30, 34))
    dr = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(tiles):
        x, y = (i % cols) * W, (i // cols) * (H + 18)
        sheet.paste(Image.fromarray(img).resize((W, H)), (x, y + 18))
        dr.text((x + 4, y + 3), name, fill=(235, 235, 235))
    sheet.save(path)


def gui_smoke(wav, flac, gl, out_dir):
    from imgui_bundle import hello_imgui, imgui

    from .modes import MODE_ORDER
    from .project import RENDER_PRESETS
    from .ui.app import StudioApp, build_runner_params

    app = StudioApp(tracks=[wav, flac], gl=gl, headless=True)
    rp = build_runner_params(app, headless=True)
    state = {'frame': 0, 'errors': [], 'visited': set()}
    orig_pre = rp.callbacks.pre_new_frame
    orig_post_init = rp.callbacks.post_init
    script = []

    def post_init():
        imgui.get_io().backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
        imgui.get_io().display_size = V(1800, 1000)
        orig_post_init()

    def V(x, y):
        return imgui.ImVec2(x, y)

    def mouse(x, y, down=None, button=0):
        io = imgui.get_io()
        io.add_mouse_pos_event(x, y)
        if down is not None:
            io.add_mouse_button_event(button, down)

    def key(k, down, ctrl=False):
        io = imgui.get_io()
        io.add_key_event(imgui.Key.mod_ctrl, ctrl)
        io.add_key_event(k, down)

    def pre():
        f = state['frame']
        state['frame'] += 1
        io = imgui.get_io()
        io.display_size = V(1800, 1000)
        try:
            # scripted interactions
            tl = app.timeline.rect
            if f == 5:
                app.seek(4.0)
            if f == 6:
                key(imgui.Key.k, True)
            if f == 7:
                key(imgui.Key.k, False)
            if 10 <= f < 10 + len(MODE_ORDER):
                k = app.current_kf()
                k.set_mode(MODE_ORDER[f - 10])
                k.use_palette = (f % 3 == 0)
                app.touch()
                state['visited'].add(MODE_ORDER[f - 10])
            g = 10 + len(MODE_ORDER)
            if f == g and tl:
                x, y, w, h = tl
                mouse(x + w * 0.3, y + 10, True)               # ruler: start scrubbing
            if f == g + 1 and tl:
                x, y, w, h = tl
                mouse(x + w * 0.6, y + 10)
            if f == g + 2 and tl:
                mouse(tl[0] + tl[2] * 0.6, tl[1] + 10, False)
                state['scrub_t'] = app.playhead
            if f == g + 4 and tl:
                # double-click the keyframe lane to add a keyframe
                x, y, w, h = tl
                mouse(x + w * 0.8, y + 22 + 13, True)
            if f == g + 5 and tl:
                mouse(tl[0] + tl[2] * 0.8, tl[1] + 35, False)
            if f == g + 6 and tl:
                mouse(tl[0] + tl[2] * 0.8, tl[1] + 35, True)
            if f == g + 7 and tl:
                mouse(tl[0] + tl[2] * 0.8, tl[1] + 35, False)
                state['kf_after_dblclick'] = len(app.project.keyframes)
            if f == g + 9 and tl:
                # drag that keyframe to the left
                k = app.project.key_by_id(app.selected_kf)
                state['drag_from'] = k.time
                kx = tl[0] + 8 + (k.time - app.timeline.t0) * app.timeline.pps
                state['kx'] = kx
                mouse(kx, tl[1] + 35, True)
            if f == g + 10 and tl:
                mouse(state['kx'] - 60, tl[1] + 35)
            if f == g + 11 and tl:
                mouse(state['kx'] - 120, tl[1] + 35)
            if f == g + 12 and tl:
                mouse(state['kx'] - 120, tl[1] + 35, False)
                state['drag_to'] = app.project.key_by_id(app.selected_kf).time
            if f == g + 14:
                key(imgui.Key.z, True, ctrl=True)
            if f == g + 15:
                key(imgui.Key.z, False, ctrl=False)
            if f == g + 17:
                kk = app.project.key_by_id(app.selected_kf)
                state['after_undo'] = kk.time if kk else None
                state['tl'] = (app.timeline.pps, app.timeline.t0, app.total(), app.timeline.rect)
            if f == g + 16:
                for name in RENDER_PRESETS:
                    app.project.render.update(RENDER_PRESETS[name])
                app.touch()
            if f == g + 18:
                app.fullscreen = True
            if f == g + 20:
                app.fullscreen = False
                app.show_help = True
            if f == g + 22:
                app.project.export.update({'width': 320, 'height': 180, 'fps': 24,
                                           'path': os.path.join(out_dir, 'gui_export.mp4')})
                app.start_export()
            if f > g + 22 and app.exporter is not None and app.exporter.done and 'export_done' not in state:
                state['export_done'] = f
            if f == g + 200 or (state.get('export_done') and f > state['export_done'] + 3):
                hello_imgui.get_runner_params().app_shall_exit = True
            orig_pre()
        except Exception as e:
            traceback.print_exc()
            state['errors'].append(f'frame {f}: {e}')
            hello_imgui.get_runner_params().app_shall_exit = True

    rp.callbacks.pre_new_frame = pre
    rp.callbacks.post_init = post_init
    orig_gui = list(rp.docking_params.dockable_windows)

    def guard(fn, name):
        def inner():
            try:
                fn()
            except Exception as e:
                traceback.print_exc()
                state['errors'].append(f'{name} frame {state["frame"]}: {e}')
                hello_imgui.get_runner_params().app_shall_exit = True
        return inner
    for w in orig_gui:
        w.gui_function = guard(w.gui_function, w.label)
    rp.docking_params.dockable_windows = orig_gui
    rp.callbacks.show_gui = guard(rp.callbacks.show_gui, 'show_gui')
    rp.callbacks.show_menus = guard(rp.callbacks.show_menus, 'menus')
    hello_imgui.run(rp)
    check('UI ran without exceptions', not state['errors'], '; '.join(state['errors'][:3]) + f' ({state["frame"]} frames)')
    check('UI visited every colour mode', len(state['visited']) == len(MODE_ORDER), f'{len(state["visited"])} modes')
    check('timeline scrub moves the playhead', state.get('scrub_t', 0) > 0.5, f'playhead {state.get("scrub_t", 0):.2f}s')
    check('double-click adds a keyframe', state.get('kf_after_dblclick', 0) >= 3, f'{state.get("kf_after_dblclick")} keyframes')
    moved = state.get('drag_to') is not None and state.get('drag_from') is not None and state['drag_to'] < state['drag_from'] - 0.2
    check('dragging a keyframe moves it', moved, f'{state.get("drag_from")} -> {state.get("drag_to")}  timeline {state.get("tl")}')
    check('undo restores the keyframe time', state.get('after_undo') is not None and abs(state['after_undo'] - state['drag_from']) < 1e-6,
          f'after undo {state.get("after_undo")}')
    gx = os.path.join(out_dir, 'gui_export.mp4')
    check('export from the UI finished', os.path.exists(gx) and state.get('export_done') is not None)
    img = app.renderer.read_rgb()
    from PIL import Image
    Image.fromarray(img).save(os.path.join(out_dir, 'gui_viewer_last_frame.png'))


if __name__ == '__main__':
    sys.exit(run())
