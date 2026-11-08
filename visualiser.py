import warnings
import sys
import os

# Suppress warnings immediately before other imports
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import pygame
import pyaudiowpatch as pyaudio
import numpy as np
import queue
import ctypes
import moderngl
import struct

# --- CONTROL VARIABLES ---
SHOW_FPS_COUNTER = False
BRIGHTNESS_MULTIPLIER = 8.0  # Increase this value to make the pixels brighter (e.g., 2.0 for double brightness)
# -------------------------

# Initialize Pygame with OpenGL
pygame.init()
pygame.font.init() # Ensure font module is ready
info = pygame.display.Info()
width, height = info.current_w, info.current_h

# Window always on top + OpenGL context
screen = pygame.display.set_mode((width, height), pygame.NOFRAME | pygame.OPENGL | pygame.DOUBLEBUF)

hwnd = pygame.display.get_wm_info()['window']
HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040

try:
    ctypes.windll.user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, 
                                      SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
except Exception as e:
    print(f"Could not set window to always on top: {e}")

clock = pygame.time.Clock()

# FPS Counter Setup
if SHOW_FPS_COUNTER:
    try:
        # Use a reliable system font
        font = pygame.font.SysFont('Consolas', 24, bold=True) 
    except pygame.error:
        font = None

# --- CONSTANTS ---
# Defined early so they are available for ModernGL setup
CHUNK = 1024
FPS = 144
min_freq = 20.0
max_freq = 40000.0
log_min_freq = np.log(min_freq)
log_scale = np.log(max_freq / min_freq)
amp_threshold = 0.01 # Match older working version
fade_factor = 0.8  # Match older working version

# --- MODERNGL SETUP ---
ctx = moderngl.create_context()
ctx.viewport = (0, 0, width, height)
ctx.enable(moderngl.BLEND)
ctx.point_size = 5.0  # Fixed visible size for all points

# Shaders for drawing the dots (with dynamic brightness)
prog_points = ctx.program(
    vertex_shader='''
        #version 330
        in vec2 in_vert; // (pan, freq_norm)
        in float in_amp;
        out float v_amp;
        
        void main() {
            // Map inputs to NDC coordinates (-1 to 1)
            float x = in_vert.x;
            float y = (in_vert.y * 2.0) - 1.0;
            
            gl_Position = vec4(x, y, 0.0, 1.0);
            v_amp = in_amp;
        }
    ''',
    fragment_shader='''
        #version 330
        in float v_amp;
        out vec4 f_color;
        
        void main() {
            f_color = vec4(v_amp, v_amp, v_amp, 1.0);
        }
    ''',
)

# Shaders for the fade effect (Texture to Screen)
prog_fade = ctx.program(
    vertex_shader='''
        #version 330
        in vec2 in_vert;
        out vec2 v_uv;
        void main() {
            v_uv = in_vert * 0.5 + 0.5;
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330
        uniform sampler2D tex;
        uniform float fade;
        in vec2 v_uv;
        out vec4 f_color;
        void main() {
            vec4 color = texture(tex, v_uv);
            f_color = vec4(color.rgb * fade, 1.0);
        }
    ''',
)
prog_fade['tex'].value = 0

# Shaders for text rendering
prog_text = ctx.program(
    vertex_shader='''
        #version 330
        in vec2 in_vert;
        in vec2 in_uv;
        out vec2 v_uv;
        void main() {
            v_uv = in_uv;
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330
        uniform sampler2D tex;
        in vec2 v_uv;
        out vec4 f_color;
        void main() {
            f_color = texture(tex, v_uv);
        }
    ''',
)
prog_text['tex'].value = 0

# Full screen quad for fading
quad_buffer = ctx.buffer(np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
     1.0,  1.0,
], dtype='f4').tobytes())
quad_vao = ctx.vertex_array(prog_fade, [(quad_buffer, '2f', 'in_vert')])

# Textures for Ping-Pong buffering (History)
tex1 = ctx.texture((width, height), 4, dtype='f4')
tex2 = ctx.texture((width, height), 4, dtype='f4')
fbo1 = ctx.framebuffer(color_attachments=[tex1])
fbo2 = ctx.framebuffer(color_attachments=[tex2])

# Initialize FBOS to BLACK
fbo1.use()
ctx.clear(0.0, 0.0, 0.0, 1.0)
fbo2.use()
ctx.clear(0.0, 0.0, 0.0, 1.0)

# State for ping-pong
active_tex = tex1
active_fbo = fbo2 # We draw FROM tex1 INTO fbo2

# Buffer for point data (Dynamic)
vbo_points = ctx.buffer(reserve=CHUNK * 3 * 4) # 3 floats * 4 bytes
vao_points = ctx.vertex_array(prog_points, [(vbo_points, '2f 1f', 'in_vert', 'in_amp')])

# --- AUDIO SETUP ---
FORMAT = pyaudio.paFloat32
p = pyaudio.PyAudio()

default_loopback = p.get_default_wasapi_loopback()
if default_loopback is None:
    print("No loopback device found. Using default input.")
    device_index = p.get_default_input_device_info()['index']
    dev = p.get_device_info_by_index(device_index)
    CHANNELS = min(2, dev['maxInputChannels'])
    RATE = int(dev['defaultSampleRate'])
else:
    device_index = default_loopback['index']
    CHANNELS = default_loopback['maxInputChannels']
    RATE = int(default_loopback['defaultSampleRate'])

audio_queue = queue.Queue(maxsize=1)

def callback(in_data, frame_count, time_info, status):
    if audio_queue.empty():
        audio_queue.put(np.frombuffer(in_data, dtype=np.float32))
    return (None, pyaudio.paContinue)

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                frames_per_buffer=CHUNK, input_device_index=device_index,
                stream_callback=callback)
stream.start_stream()

running = True

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q)):
                running = False

        try:
            data = audio_queue.get_nowait()
        except queue.Empty:
            data = None

        points_data = b'' # Bytes to send to GPU
        num_points = 0

        if data is not None:
            data = data.copy()

            if CHANNELS >= 2:
                dataL = data[0::2]
                dataR = data[1::2]
            else:
                dataL = dataR = data

            hanning = np.hanning(len(dataL))
            dataL *= hanning
            dataR *= hanning

            fftL = np.abs(np.fft.rfft(dataL))
            fftR = np.abs(np.fft.rfft(dataR))
            freqs = np.fft.rfftfreq(CHUNK, 1.0 / RATE)

            max_amp = np.max(np.concatenate((fftL, fftR))) + 1e-6

            valid_indices = np.where(freqs[1:] > min_freq)[0] + 1
            
            if len(valid_indices) > 0:
                v_freqs = freqs[valid_indices]
                v_fftL = fftL[valid_indices]
                v_fftR = fftR[valid_indices]
                
                amps = (v_fftL + v_fftR) / 2.0
                amp_mask = amps > (amp_threshold * max_amp)
                
                if np.any(amp_mask):
                    final_amps = amps[amp_mask]
                    final_fftL = v_fftL[amp_mask]
                    final_fftR = v_fftR[amp_mask]
                    final_freqs = v_freqs[amp_mask]

                    if CHANNELS >= 2:
                        pan = (final_fftR - final_fftL) / (final_fftR + final_fftL + 1e-6)
                    else:
                        pan = np.zeros_like(final_amps)

                    # Prepare data for GPU
                    norm_freqs = np.log(final_freqs / min_freq) / log_scale
                    
                    # Normalize Amplitude and apply brightness multiplier
                    norm_amps = (final_amps / max_amp) * BRIGHTNESS_MULTIPLIER
                    
                    gpu_data = np.column_stack((pan, norm_freqs, norm_amps)).astype('f4')
                    points_data = gpu_data.tobytes()
                    num_points = len(final_amps)

        # --- RENDER PASS ---
        
        # 1. Bind the target framebuffer (The "Next" frame)
        active_fbo.use()
        
        # 2. Draw the PREVIOUS frame into the CURRENT frame with fade
        ctx.disable(moderngl.BLEND) 
        active_tex.use(location=0) # Bind previous texture
        prog_fade['fade'].value = fade_factor
        quad_vao.render(moderngl.TRIANGLE_STRIP)
        
        # 3. Draw the NEW points on top with additive blending
        if num_points > 0:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            vbo_points.write(points_data)
            vao_points.render(moderngl.POINTS, vertices=num_points)

        # 4. Display to Screen
        ctx.screen.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(moderngl.BLEND) 
        
        # Display the FBO we just drew into
        if active_fbo == fbo1:
            tex1.use(location=0)
        else:
            tex2.use(location=0)
            
        prog_fade['fade'].value = 1.0
        quad_vao.render(moderngl.TRIANGLE_STRIP)

        # 5. Render FPS counter if enabled
        if SHOW_FPS_COUNTER and font:
            fps = clock.get_fps()
            fps_text = font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
            tw, th = fps_text.get_size()

            left = -1.0 + (20.0 / width)
            right = left + (2.0 * tw / width)
            top = 1.0 - (20.0 / height)
            bottom = top - (2.0 * th / height)

            vert_data = np.array([
                left, bottom, 0.0, 1.0,  # Flip UV y because pygame tostring is top-down, but GL bottom-up
                right, bottom, 1.0, 1.0,
                left, top, 0.0, 0.0,
                right, top, 1.0, 0.0,
            ], dtype='f4')

            text_data = pygame.image.tostring(fps_text, 'RGBA', False)

            if 'text_tex' not in globals() or text_tex.size != (tw, th):
                if 'text_tex' in globals():
                    text_tex.release()
                text_tex = ctx.texture((tw, th), 4, text_data)
                text_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

                if 'text_vbo' in globals():
                    text_vbo.release()
                text_vbo = ctx.buffer(vert_data.tobytes())
                if 'text_vao' in globals():
                    text_vao.release()
                text_vao = ctx.vertex_array(prog_text, [(text_vbo, '2f 2f', 'in_vert', 'in_uv')])
            else:
                text_tex.write(text_data)
                text_vbo.write(vert_data.tobytes())

            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            text_tex.use(0)
            text_vao.render(moderngl.TRIANGLE_STRIP)
            ctx.disable(moderngl.BLEND)

        # 6. Swap Buffers for next frame
        if active_fbo == fbo1:
            active_fbo = fbo2
            active_tex = tex1
        else:
            active_fbo = fbo1
            active_tex = tex2

        pygame.display.flip()
        clock.tick(FPS)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'stream' in locals() and stream.is_active():
        stream.stop_stream()
    if 'stream' in locals():
        stream.close()
    if 'p' in locals():
        p.terminate()
    pygame.quit()