import librosa
import numpy as np
import time
import threading
import queue
import pickle
import matplotlib.pyplot as plt

# Desired sample rate 16000 Hz
sample_rate = 16000
# Frame length
frame_length = 512

# Path to the audio file
audio_file = "audio_aaico_challenge.wav"

# Read the audio file and resample it to the desired sample rate
audio_data, _ = librosa.load(audio_file, sr=sample_rate)

# Create a list to store the labeled samples
labeled_samples = []

# Simulate the streaming and labeling process
for i in range(0, len(audio_data), frame_length):
    # Simulate real-time processing
    time.sleep(frame_length / sample_rate)

    # Simulate extracting features from the audio frame
    frame_audio_data = audio_data[i:i + frame_length]

    # Append the labeled sample to the list (for now, we label everything as a command)
    labeled_samples.append((frame_audio_data, 1))

# Save the labeled samples to a file
with open('label_samples.pkl', 'wb') as file:
    pickle.dump(labeled_samples, file)

# # Print out the labeled samples
# for sample in labeled_samples:
#     print("Label:", sample[1])  # Print the label
#     plt.plot(sample[0])  # Plot the audio data
#     plt.show()
