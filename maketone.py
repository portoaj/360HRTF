# This script creates a simple tone and saves it to a wav using numpy and soundfile
import numpy as np
import soundfile as sf

OUTPUT_FILE = 'tone.wav'
DURATION = 60 # duration in seconds
VOLUME = 0.01
sr = 44100


data = np.random.uniform(-1 * VOLUME, 1 * VOLUME, sr * DURATION)

sf.write(OUTPUT_FILE, data, sr)