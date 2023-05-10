import soundfile as sf
import numpy as np
import sounddevice as sd


# This script moves a sound source straight to the right
INPUT_FILE = 'furelise.mp3'
OUTPUT_FILE = 'rightfurelise.wav'
HRIR_FILE = 'kemardb\elev0\H0e090a.wav'

# Load rightward HRIR
HRIR, HRIR_sr = sf.read(HRIR_FILE)
print('HRIR shape:', HRIR.shape, 'HRIR Sample Rate:', HRIR_sr)

# Load input audio file
input_audio, input_audio_sr = sf.read(INPUT_FILE)
input_audio = np.array(input_audio)
# Make input audio mono if it's stereo
if len(input_audio.shape) > 1 and input_audio.shape[1] > 1:
    input_audio = np.mean(input_audio, axis=1)
print('Input audio shape:', input_audio.shape, 'Input audio sample rate:', input_audio_sr)


# Convolve audio with rightward HRIR
left_result = np.convolve(input_audio, HRIR[:, 0])
right_result = np.convolve(input_audio, HRIR[:, 1])
result = np.vstack([left_result, right_result]).transpose()

sf.write(OUTPUT_FILE, result, HRIR_sr)
sd.play(result, HRIR_sr, blocking=True, loop=False)
