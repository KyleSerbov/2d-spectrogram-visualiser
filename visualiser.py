import pygame
import pygame.pixelcopy
import pyaudiowpatch as pyaudio
import numpy as np
import queue
import ctypes
import warnings

# Suppress the pkg_resources warning from Pygame
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# Initialize Pygame
pygame.init()
info = pygame.display.Info()
width, height = info.current_w, info.current_h

# Window always on top
screen = pygame.display.set_mode((width, height), pygame.NOFRAME) 

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

try:
    font = pygame.font.Font(None, 36) 
except pygame.error:
    font = None

# Audio parameters
CHUNK = 2048
FORMAT = pyaudio.paFloat32
p = pyaudio.PyAudio()

# Loopback device setup
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
        # This creates a read-only numpy array from the raw bytes
        audio_queue.put(np.frombuffer(in_data, dtype=np.float32))
    return (None, pyaudio.paContinue)

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                frames_per_buffer=CHUNK, input_device_index=device_index,
                stream_callback=callback)
stream.start_stream()

# Ghosting parameters
FPS = 144
fade_factor = 0.8
intensity = np.zeros((height, width), dtype=np.float64)

min_freq = 10.0
max_freq = 40000.0

# Pre-calculate Numpy Kernel
spot_radius = 5
ky, kx = np.ogrid[-spot_radius:spot_radius+1, -spot_radius:spot_radius+1]
mask = kx**2 + ky**2 <= spot_radius**2
kernel_offsets_y, kernel_offsets_x = np.where(mask)
kernel_offsets_y -= spot_radius
kernel_offsets_x -= spot_radius

log_min_freq = np.log(min_freq)
log_scale = np.log(max_freq / min_freq)

amp_threshold = 0.01

running = True

display_surface = pygame.Surface((width, height))

# Buffer to store RGB data before copying to surface
rgb_buffer = np.empty((width, height, 3), dtype=np.uint8)

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q)):
                running = False

        intensity *= fade_factor

        try:
            data = audio_queue.get_nowait()
        except queue.Empty:
            data = None

        if data is not None:
            # FIX: Create a writable copy of the read-only buffer
            data = data.copy()

            if CHANNELS >= 2:
                dataL = data[0::2]
                dataR = data[1::2]
            else:
                dataL = dataR = data

            # Now this in-place multiplication works because data is writable
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
                        
                    xs = ((pan + 1.0) / 2.0 * (width - 1)).astype(int)
                    
                    logf = np.log(final_freqs / min_freq) / log_scale
                    ys = (height - 1 - (logf * (height - 1))).astype(int)
                    
                    add_amps = (final_amps / max_amp) * 255.0

                    target_ys = ys[:, np.newaxis] + kernel_offsets_y
                    target_xs = xs[:, np.newaxis] + kernel_offsets_x
                    
                    flat_ys = target_ys.ravel()
                    flat_xs = target_xs.ravel()
                    flat_amps = np.repeat(add_amps, len(kernel_offsets_y))

                    valid_pixels = (flat_ys >= 0) & (flat_ys < height) & \
                                   (flat_xs >= 0) & (flat_xs < width)
                    
                    np.add.at(intensity, 
                              (flat_ys[valid_pixels], flat_xs[valid_pixels]), 
                              flat_amps[valid_pixels])

        intensity = np.clip(intensity, 0, 255)

        intensity_uint8 = intensity.T.astype(np.uint8)
        
        rgb_buffer[:, :, 0] = intensity_uint8
        rgb_buffer[:, :, 1] = intensity_uint8
        rgb_buffer[:, :, 2] = intensity_uint8
        
        pygame.pixelcopy.array_to_surface(display_surface, rgb_buffer)
        
        screen.blit(display_surface, (0, 0))
        
        #if font:
        #    fps = clock.get_fps()
        #    fps_text = font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
        #    screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

except Exception as e:
    print(f"Error: {e}")

finally:
    if stream.is_active():
        stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()