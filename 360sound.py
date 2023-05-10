import soundfile as sf
import numpy as np
import os
import tqdm
import time

# This script moves a sound source in a circle around the listener

INPUT_FILE = 'tone.wav'
OUTPUT_FILE = '360tone.wav'
HRIR_FOLDER = 'kemardb\elev0'

CHUNK_LENGTH = 500 # Duration in ms for each discrete HRTF

# Get all HRIR filepaths
HRIR_FILES = [file for file in os.listdir(HRIR_FOLDER)]

# Load input audio file
input_audio, input_audio_sr = sf.read(INPUT_FILE)
input_audio = np.array(input_audio)

# Make input audio mono if it's stereo
if len(input_audio.shape) > 1 and input_audio.shape[1] > 1:
    input_audio = np.mean(input_audio, axis=1)
print('Input audio shape:', input_audio.shape, 'Input audio sample rate:', input_audio_sr)

# Get song length in ms
input_duration_ms = input_audio.shape[0] * 1000 // input_audio_sr
num_chunks = input_duration_ms // CHUNK_LENGTH
samples_per_chunk = int((CHUNK_LENGTH)/ 1000 * input_audio_sr)
print('Input duration in ms', input_duration_ms, 'Num chunks in song', num_chunks)

results = None
for chunk in tqdm.tqdm(range(num_chunks)):
    # Beginning and ending samples
    chunk_start_samples = chunk * samples_per_chunk
    chunk_end_samples = (chunk + 1) * samples_per_chunk

    # Get portion of audio for this chunk
    input_chunk = input_audio[chunk_start_samples: chunk_end_samples]
    
    # Find appropriate HRIR for this chunk
    CHUNK_INDEX = chunk % (len(HRIR_FILES) * 2)
    if CHUNK_INDEX < len(HRIR_FILES):
        HRIR_file = os.path.join(HRIR_FOLDER, HRIR_FILES[chunk % len(HRIR_FILES)])
    else:
        HRIR_file = os.path.join(HRIR_FOLDER, HRIR_FILES[len(HRIR_FILES) - chunk % len(HRIR_FILES) - 2])

    # Find HRIR azimuth angle
    azimuth = CHUNK_INDEX * 5
    print(azimuth, HRIR_file, azimuth <= 180)

    # Load HRIR file
    HRIR, HRIR_sr = sf.read(HRIR_file)
    left_result = np.convolve(input_chunk, HRIR[:, 0], mode='same')
    right_result = np.convolve(input_chunk, HRIR[:, 1], mode='same')
    if azimuth <= 180:
        result = np.vstack([left_result, right_result]).transpose()
    else:
        result = np.vstack([right_result, left_result]).transpose()

    # Add this result to results array
    if results is None:
        results = result
    else:
        results = np.concatenate((results, result), axis=0)
print(results.shape)

#HRIR, HRIR_sr = sf.read(HRIR_FILE)
#print('HRIR shape:', HRIR.shape, 'HRIR Sample Rate:', HRIR_sr)


# Convolve audio with rightward HRIR
#left_result = np.convolve(input_audio, HRIR[:, 0])
#right_result = np.convolve(input_audio, HRIR[:, 1])
#result = np.vstack([left_result, right_result]).transpose()

sf.write(OUTPUT_FILE, results, HRIR_sr)

    
#import sounddevice as sd
#sd.play(results, HRIR_sr, blocking=False, loop=False)

#for chunk in range(num_chunks):
    # Find appropriate HRIR for this chunk
 #   print(HRIR_FILES[chunk % len(HRIR_FILES)])
  #  time.sleep(CHUNK_LENGTH/ 1000)
#sd.stop()


#print(results.shape)