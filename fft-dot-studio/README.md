# FFT Dot Studio

An editor for the 2D FFT-dot visualiser that works like a video editor. You import a track or a whole
album, and the dots appear the way they did in v5/v6: pan on x, log-frequency on y, additive glow and a
fading trail. You can put a keyframe at any point in the timeline with a colour mode, a palette and
parameters, and the app tweens between them. When you're done, it exports a frame-exact video with the
audio muxed in.

## Run

```
python studio.py                         # empty project
python studio.py song.flac other.mp3     # open with tracks
python studio.py album.fdsproj           # open a project
python studio.py --selftest              # headless tests (no window, no sound), ~1 min
python studio.py --export album.fdsproj out.mp4 [--width 3840 --height 2160 --fps 60]
```

The window opens maximised on the smaller monitor.

Requirements (installed): `imgui-bundle moderngl glfw numpy scipy soundfile sounddevice torch
torchaudio demucs pillow` and `ffmpeg` on PATH or via winget. The GPU (CUDA through torch) runs the STFT,
the f0 search and Demucs.

Video export encodes on the GPU (NVENC). ffmpeg 9.x needs NVIDIA driver 610+, so the app prefers a bundled
ffmpeg 8.1 at `tools/ffmpeg/bin/ffmpeg.exe` (not in git; BtbN `ffmpeg-n8.1-latest-win64-gpl-8.1.zip`, extracted
to `tools/ffmpeg`), or `FDS_FFMPEG` if set. Without it, export falls back to libx264 on the CPU.

## Workflow

1. **Project panel**: add audio. FLAC at any bit depth, sample rate or channel count is supported, plus
   WAV, MP3, OGG, M4A and others. Several files play back to back as an album. Analysis runs in the
   background: the spectrum is ready in a second or two, then features, Demucs stems and vowels follow.
2. **Timeline**:
   - Drag the ruler to scrub.
   - Double-click the keyframe lane to add a keyframe. Drag keyframes to move them (hold Shift to snap
     to beats). Right-click a keyframe to delete, duplicate or change its easing.
   - The lanes show waveforms, track boundaries, beats, the colour blend between keyframes, and the
     detected vowels.
3. **Keyframe panel**: choose a mode, its parameters and a palette. Use the gradient editor to change
   colours, or tick "Use palette" to recolour a native-colour mode. Each keyframe can also animate
   render settings such as brightness, dot size, trail, glow and background.
4. **Render / Analysis panels**: FFT and look settings, with presets "Classic v5", "Punchy beats",
   "Smooth bass" and more. Every setting has a tooltip.
5. **Export**: the frame rate, resolution and codec are configurable. Frame k shows t = k / fps
   exactly, and the audio track has the same length and starts at 0.

## Colour modes

Ported from earlier versions:

| Mode | Source version |
|---|---|
| Mono White | v1–v5 |
| Note Rainbow | v6 rainbow |
| Note Hue Continuous | portfolio |
| Note Regions | v6 notes / v7 |
| Note Regions Blend | v6 notes-blending |
| Peaks Only Rainbow | rainbow-improved |
| Peak Halo | rainbow-improved-blending |
| Split Bass Rainbow / Treble Regions | blending-update |
| Peak Sparkle | update-2, first GitHub version |
| Beat Zones | v7 |

New modes: Radial Glow (centre colour fading outwards, with the centre at the screen centre, the live
dot centroid or a custom point), Polar Swirl, Amplitude Heat, Frequency Gradient, Stereo Field,
Harmonic Series, Onset Flash, Timbre, Key / Chord Colour, Consonance, Time Cycle, Flow Field Marble,
Beat Pulse, Attack / Decay, Cluster Density, Instrument Stems (Demucs), Vowel / Mouth Shape,
Note + Octave, Pitch Spiral, and Solid Colour.

Between keyframes, the same mode **morphs** (parameters and palettes are interpolated). Different modes
**crossfade** (each dot's colour is blended in OKLab). Easings: linear, smooth, ease in/out/in-out,
exponential, hold.

## Timing

- **Frame alignment** (Analysis panel; applies to the preview and to exports):
  - *Centred* (default): the window is centred on t, so a hit peaks exactly when it happens.
  - *Live-style*: the window ends at t, as in the real-time v5.
  - *Custom*: a ± ms offset.

  Multi-resolution bands are each aligned for their own window size.
- **Output latency** (transport bar; preview only, never applied to exports): the dots follow the
  audio device clock minus the reported device latency, plus your manual extra ms.

## Shortcuts

| Key | Action |
|---|---|
| Space | play / pause |
| ← / → | step 1/30 s (Shift: 1 s) |
| Ctrl+← / → | previous / next keyframe |
| Home / End | jump to start / end |
| K | add a keyframe at the playhead |
| Ctrl+D | duplicate the keyframe |
| Del | delete the keyframe |
| Ctrl+Z / Ctrl+Y | undo / redo |
| Ctrl+S / Ctrl+Shift+S | save / save as |
| Ctrl+O / Ctrl+N | open / new |
| Ctrl+I | add audio |
| Ctrl+E | export |
| F | fullscreen preview (Esc to leave) |
| M | mute |
| F1 | help |

In the timeline, Ctrl+wheel zooms and the wheel scrolls. In the palette bar, double-click to add a
stop, drag a stop to move it, and right-click a stop to delete it.

## Cache

The analysis is cached in `%LOCALAPPDATA%\fft-dot-studio\cache`, which you can delete at any time.
Custom palettes are saved in `%LOCALAPPDATA%\fft-dot-studio\palettes.json`.
