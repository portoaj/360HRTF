import soundfile as sf
import numpy as np
import os
import tqdm
import time

# This script moves a sound source in a circle around the listener

INPUT_FILE = 'samples/tone.wav'
OUTPUT_FILE = '180tone.wav'
HRIR_FOLDER = 'kemardb\elev0'

BUFFER_LENGTH = 11025 # Number of samples in buffer

# Get all HRIR filepaths
HRIR_FILES = [file for file in os.listdir(HRIR_FOLDER)]
HRIR_FILES.sort()

# Load input audio file
input_audio, input_audio_sr = sf.read(INPUT_FILE)
input_audio = np.array(input_audio)

# Make input audio mono if it's stereo
if len(input_audio.shape) > 1 and input_audio.shape[1] > 1:
    input_audio = np.mean(input_audio, axis=1)
print('Input audio shape:', input_audio.shape, 'Input audio sample rate:', input_audio_sr)

# Get song length in ms
num_samples = input_audio.shape[0]
input_duration_ms = num_samples * 1000 // input_audio_sr


print('Input duration in ms', input_duration_ms)

results = np.zeros((num_samples + (BUFFER_LENGTH - num_samples % BUFFER_LENGTH), 2))
previous_overlap = None
for starting_sample in tqdm.tqdm(range(0, num_samples, BUFFER_LENGTH)):
    # Get portion of audio for this buffer
    input_chunk = input_audio[starting_sample: starting_sample + BUFFER_LENGTH]

    # Find appropriate HRIR for this chunk
    CHUNK_INDEX = starting_sample // BUFFER_LENGTH % len(HRIR_FILES)
    HRIR_file = os.path.join(HRIR_FOLDER, HRIR_FILES[CHUNK_INDEX])

    # Load HRIR file
    HRIR, HRIR_sr = sf.read(HRIR_file)

    # Find padding length needed to bring each convolution result to 2 * BUFFER_LENGTH size
    padding_amount = 2 * BUFFER_LENGTH - (len(input_chunk) + len(HRIR[:, 0]) - 1)

    # Convolve the input with the HRIR and pad zeros
    left_result = np.convolve(input_chunk, HRIR[:, 0])
    right_result = np.convolve(input_chunk, HRIR[:, 1])

    left_result = np.pad(left_result, ((0, padding_amount)), mode='constant')
    right_result = np.pad(right_result, ((0, padding_amount)), mode='constant')

    # Format the result into a shape of (BUFFER_LENGTH, 2) and add overlap if applicable
    result = np.vstack([left_result[0: BUFFER_LENGTH], right_result[0: BUFFER_LENGTH]]).transpose()

    if previous_overlap is not None:
       result = np.add(result, previous_overlap)

    previous_overlap = np.vstack([left_result[BUFFER_LENGTH:], right_result[BUFFER_LENGTH:]]).transpose()
    #print(result, previous_overlap, result.shape, previous_overlap.shape)

    results[starting_sample: starting_sample + BUFFER_LENGTH] = result

print(results.shape)

sf.write(OUTPUT_FILE, results, HRIR_sr)
