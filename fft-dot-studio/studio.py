"""FFT Dot Studio - keyframed colour editor for the 2D FFT-dot spectrogram visualiser.

  python studio.py [project.fdsproj] [audio files...]     open the editor
  python studio.py --selftest [--quick] [--out DIR]       headless tests (no window, no sound)
  python studio.py --export project.fdsproj out.mp4 [--width W --height H --fps N]
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore', message='.*HF_TOKEN.*')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')


def main():
    ap = argparse.ArgumentParser(description='FFT Dot Studio')
    ap.add_argument('files', nargs='*', help='a .fdsproj project and/or audio files')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--quick', action='store_true', help='shorter self-test')
    ap.add_argument('--no-gui-test', action='store_true')
    ap.add_argument('--out', default=None, help='self-test output folder')
    ap.add_argument('--export', nargs=2, metavar=('PROJECT', 'OUTPUT'))
    ap.add_argument('--width', type=int)
    ap.add_argument('--height', type=int)
    ap.add_argument('--fps', type=int)
    args = ap.parse_args()

    if args.selftest:
        from fds.selftest import run
        sys.exit(run(args.out, quick=args.quick, gui=not args.no_gui_test))

    if args.export:
        import moderngl
        from fds.analysis import AnalysisManager
        from fds.engine import Engine
        from fds.export import Exporter, probe
        from fds.project import Project
        proj = Project.load(args.export[0])
        ex = proj.export
        mgr = AnalysisManager(use_demucs=proj.use_demucs, sync=True, vocal_source=proj.vocal_source)
        mgr.set_settings(proj.analysis)
        eng = Engine(proj, mgr)
        for t in proj.tracks:
            print('analysing', os.path.basename(t))
            mgr.get(t)
        gl = moderngl.create_standalone_context()
        e = Exporter(eng, gl, args.export[1], args.width or ex['width'], args.height or ex['height'],
                     args.fps or ex['fps'], ex['codec'], ex['audio'], ex['quality'])

        def prog(p):
            print(f'\r{p * 100:5.1f}%', end='', flush=True)
        e.run_blocking(prog)
        print('\n', probe(args.export[1]))
        return

    project = next((f for f in args.files if f.lower().endswith(('.fdsproj', '.json'))), None)
    tracks = [f for f in args.files if f != project]
    from fds.ui.app import run
    run(project, tracks)


if __name__ == '__main__':
    main()
