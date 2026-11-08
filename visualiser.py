import pygame
import pyaudiowpatch as pyaudio
import numpy as np
from math import log
import queue
import ctypes

# Initialize Pygame
pygame.init()
info = pygame.display.Info()
width, height = info.current_w, info.current_h

# --- FIX 1: WINDOW ALWAYS ON TOP ---
# Use NOFRAME for a borderless window, then force it to the top using Windows API
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

# --- NEW: FPS Counter Setup ---
try:
    # Use default system font
    font = pygame.font.Font(None, 36) 
except pygame.error:
    font = None
# -----------------------------

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

def freq_to_y(freq):
    if freq <= min_freq:
        return height - 1
    logf = log(freq / min_freq) / log(max_freq / min_freq)
    return height - 1 - int(logf * (height - 1))

amp_threshold = 0.01
spot_radius = 5

running = True

# --- FIX 2: WRAP IN TRY/FINALLY FOR CLEAN EXIT ---
try:
    while running:
        for event in pygame.event.get():
            # Added K_q to the quit keys
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q)):
                running = False

        intensity *= fade_factor

        try:
            data = audio_queue.get_nowait()
        except queue.Empty:
            data = None

        if data is not None:
            if CHANNELS >= 2:
                dataL = data[0::2]
                dataR = data[1::2]
            else:
                dataL = dataR = data

            hanningL = np.hanning(len(dataL))
            hanningR = np.hanning(len(dataR))
            dataL = dataL * hanningL
            dataR = dataR * hanningR

            fftL = np.abs(np.fft.rfft(dataL))
            fftR = np.abs(np.fft.rfft(dataR))
            freqs = np.fft.rfftfreq(CHUNK, 1.0 / RATE)

            max_amp = np.max(np.concatenate((fftL, fftR))) + 1e-6

            # --- VISUALS REVERTED TO DOTS ---
            for i in range(1, len(freqs)):
                ampL = fftL[i]
                ampR = fftR[i]
                amp = (ampL + ampR) / 2.0
                if amp > amp_threshold * max_amp:
                    pan = (ampR - ampL) / (ampR + ampL + 1e-6) if CHANNELS >= 2 else 0.0
                    x = int((pan + 1.0) / 2.0 * (width - 1))
                    y = freq_to_y(freqs[i])
                    
                    add_amp = (amp / max_amp) * 255.0
                    
                    # Original dot drawing logic
                    for dy in range(-spot_radius, spot_radius + 1):
                        for dx in range(-spot_radius, spot_radius + 1):
                            if (dx**2 + dy**2) <= spot_radius**2:
                                yy = min(max(y + dy, 0), height - 1)
                                xx = min(max(x + dx, 0), width - 1)
                                intensity[yy, xx] += add_amp

        intensity = np.clip(intensity, 0, 255)

        img = np.repeat(intensity[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
        
        # --- NEW: Draw FPS counter ---
        if font:
            fps = clock.get_fps()
            # Render the text (Text, Antialias=True, Color=White)
            fps_text = font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
            screen.blit(fps_text, (10, 10)) # Draw at (10, 10)
        # -----------------------------

        pygame.display.flip()
        clock.tick(144)

except Exception as e:
    print(f"Error: {e}")

finally:
    # This block ensures the program quits cleanly without freezing
    if stream.is_active():
        stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()