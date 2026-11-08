import pygame
import pyaudiowpatch as pyaudio
import numpy as np
from math import log
import queue

# Initialize Pygame
pygame.init()
info = pygame.display.Info()
width, height = info.current_w, info.current_h
screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
clock = pygame.time.Clock()

# Audio parameters
CHUNK = 2048  # Buffer size; adjust for latency vs. frequency resolution
FORMAT = pyaudio.paFloat32  # Float for easier processing

# Initialize PyAudioWPatch
p = pyaudio.PyAudio()

# Get default WASAPI loopback device for speakers
default_loopback = p.get_default_wasapi_loopback()
if default_loopback is None:
    print("No loopback device found. Ensure speakers are active and try enabling 'Stereo Mix' in sound settings as fallback.")
    # Fallback to default input (e.g., microphone)
    device_index = p.get_default_input_device_info()['index']
    dev = p.get_device_info_by_index(device_index)
    CHANNELS = min(2, dev['maxInputChannels'])
    RATE = int(dev['defaultSampleRate'])
else:
    device_index = default_loopback['index']
    CHANNELS = default_loopback['maxInputChannels']
    RATE = int(default_loopback['defaultSampleRate'])

if CHANNELS < 2:
    print("Warning: Using mono input (1 channel). Direction detection will be disabled (everything centered).")

# Setup audio queue
audio_queue = queue.Queue(maxsize=1)  # Keep only the latest chunk

def callback(in_data, frame_count, time_info, status):
    audio_queue.put(np.frombuffer(in_data, dtype=np.float32))
    return (None, pyaudio.paContinue)

# Open audio stream with callback
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                frames_per_buffer=CHUNK, input_device_index=device_index,
                stream_callback=callback)
stream.start_stream()

# Ghosting parameters
FPS = 144
fade_factor = 0.8
intensity = np.zeros((height, width), dtype=np.float64)

# Frequency mapping (log scale, high freq at top, low at bottom)
min_freq = 10.0
max_freq = 40000.0

def freq_to_y(freq):
    if freq <= min_freq:
        return height - 1
    logf = log(freq / min_freq) / log(max_freq / min_freq)
    return height - 1 - int(logf * (height - 1))  # Invert for high at top (y=0)

# Threshold and spot size
amp_threshold = 0.01  # Relative to max amp
spot_radius = 5  # Pixels for the 'spot' to make it visible

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    # Fade the intensity for ghosting
    intensity *= fade_factor

    # Get latest audio data if available (non-blocking)
    try:
        data = audio_queue.get_nowait()
    except queue.Empty:
        data = None  # No new data; skip processing or use previous?

    if data is not None:
        if CHANNELS >= 2:
            dataL = data[0::2]
            dataR = data[1::2]
        else:
            dataL = dataR = data  # Mono: treat as center

        # Apply Hanning window for smoother FFT (avoid in-place)
        hanningL = np.hanning(len(dataL))
        hanningR = np.hanning(len(dataR))
        dataL = dataL * hanningL
        dataR = dataR * hanningR

        # Compute FFT (real part only)
        fftL = np.abs(np.fft.rfft(dataL))
        fftR = np.abs(np.fft.rfft(dataR))
        freqs = np.fft.rfftfreq(CHUNK, 1.0 / RATE)

        # Find max amp for normalization
        max_amp = np.max(np.concatenate((fftL, fftR))) + 1e-6

        # Process each frequency bin (skip DC at i=0)
        for i in range(1, len(freqs)):
            ampL = fftL[i]
            ampR = fftR[i]
            amp = (ampL + ampR) / 2.0
            if amp > amp_threshold * max_amp:
                # Compute pan (-1 left to 1 right); 0 if not stereo
                pan = (ampR - ampL) / (ampR + ampL + 1e-6) if CHANNELS >= 2 else 0.0
                # Map to x position
                x = int((pan + 1.0) / 2.0 * (width - 1))
                # Map freq to y
                y = freq_to_y(freqs[i])
                # Add intensity as a circular spot
                add_amp = (amp / max_amp) * 255.0
                for dy in range(-spot_radius, spot_radius + 1):
                    for dx in range(-spot_radius, spot_radius + 1):
                        if (dx**2 + dy**2) <= spot_radius**2:
                            yy = min(max(y + dy, 0), height - 1)
                            xx = min(max(x + dx, 0), width - 1)
                            intensity[yy, xx] += add_amp

    # Clip intensity to 0-255
    intensity = np.clip(intensity, 0, 255)

    # Convert to RGB image (grayscale)
    img = np.repeat(intensity[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    # Create Pygame surface (swap axes for correct orientation)
    surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    # Blit to screen
    screen.blit(surface, (0, 0))

    # Update display
    pygame.display.flip()

    # Limit to 144 FPS
    clock.tick(144)

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()