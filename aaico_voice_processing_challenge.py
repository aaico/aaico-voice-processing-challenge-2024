import librosa
import numpy as np
import time
import threading
import queue
import pickle

########### PARAMETERS ###########
# Desired sample rate 16000 Hz
sample_rate = 16000
# Frame length
frame_length = 512



########### AUDIO FILE ###########
# Path to the audio file
audio_file = "test_aaico_challenge.wav"

# Read the audio file and resample it to the desired sample rate
audio_data, current_sample_rate = librosa.load(
    audio_file, 
    sr=sample_rate,
)
audio_data_int16 = (audio_data * 32767).astype(np.int16)
number_of_frames = len(audio_data_int16) // frame_length
audio_data_int16 = audio_data_int16[:number_of_frames * frame_length]
audio_duration = len(audio_data_int16) / sample_rate



########### STREAMING SIMULATION ###########
detection_mask = np.zeros(len(audio_data_int16), dtype=np.int16) # set to 1 if broadcast
buffer = queue.Queue()

def emit_data():
    t_f_s = time.time_ns()
    time_measurement.append(t_f_s)
    print('Start emitting')
    for i in range(0, number_of_frames):
        time.sleep(frame_length / sample_rate) # Simulate real time
        frame = audio_data_int16[i*frame_length: (i+1)*frame_length]
        buffer.put(frame)
    print('Stop emitting')

def process_data():
    i = 0
    print('Start processing')
    while i != number_of_frames:
        frame = buffer.get()
        
        ### YOUR CODE ###
        
        i += 1
    print('Stop processing')
    t_f_e = time.time_ns()
    time_measurement.append(t_f_e)
    # Save the list to a file
    with open('results.pkl', 'wb') as file:
        pickle.dump([time_measurement, detection_mask], file)



if __name__ == "__main__":
    time_measurement = []

    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)

    thread_process.start()
    thread_emit.start()