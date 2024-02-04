import librosa
import numpy as np
import time
import threading
import queue
import pickle
import pvporcupine

import pygame

porcupine = pvporcupine.create(
  access_key='zhpWuhGi8aOj2ssRk/jqEG1Vj96CjTaXRynJmYlgTQdC9lkPDjINbQ==',
#   keywords=['GALACTIC-BATTERY', 'LACTIC-OXYGEN', 'LACTIC-TEMPERATURE'],
  keyword_paths=['GALACTIC-BATTERY_en_mac_v3_0_0.ppn', 'LACTIC-OXYGEN_en_mac_v3_0_0.ppn', 'LACTIC-TEMPERATURE_en_mac_v3_0_0.ppn'],
)

# print(porcupine.sample_rate)
# print(porcupine.frame_length)



########### PARAMETERS ###########
# DO NOT MODIFY
# Desired sample rate 16000 Hz
# sample_rate = 16000
sample_rate = 16000
# Frame length
frame_length = 512



########### AUDIO FILE ###########
# DO NOT MODIFY
# Path to the audio file
audio_file = "audio_aaico_challenge.wav"

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
# DO NOT MODIFY
results = np.zeros(shape=(3, len(audio_data_int16)), dtype=np.int64)
# Detection mask lines are SENT TIME, LABEL, RECEIVE TIME. 
buffer = queue.Queue()
start_event = threading.Event()

def label_samples(list_samples_id, labels):
    receive_time = time.time_ns()
    results[1][list_samples_id] = labels
    results[2][list_samples_id] = receive_time

def notice_send_samples(list_samples_id):
    send_time = time.time_ns()
    results[0][list_samples_id] = send_time

def emit_data(): 
    time.sleep(.5)
    print('Start emitting')
    start_event.set()
    for i in range(0, number_of_frames):
        list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
        time.sleep(frame_length / sample_rate) # Simulate real time
        frame = audio_data_int16[list_samples_id]
        buffer.put(frame)
        notice_send_samples(list_samples_id)
    print('Stop emitting')

def process_data():
    i = 0
    start_event.wait()
    print('Start processing')
    
    
    # pygame.mixer.pre_init(frequency=sample_rate)
    # # Create a Pygame mixer Sound object
    # sound = pygame.mixer.Sound(buffer=np.zeros((frame_length, 2), dtype=np.int16))


    while i != number_of_frames:
        frame = buffer.get()
        keyword_index = porcupine.process(frame)

    #     # If the frame is mono, duplicate it to create stereo audio
    #     if frame.shape[0] == frame_length:
    #         frame2 = np.column_stack((frame, frame))

    #     # Set the sound buffer by updating the array data
    #     sound_array = pygame.sndarray.samples(sound)
    #     sound_array[:len(frame2), :] = frame2  # Assuming stereo audio

    #     # Play the audio frame using Pygame
    #     sound.play()

        ### TODO: YOUR CODE
        # MODIFY

        # print(keyword_index)

        if keyword_index >= 0:
            print(keyword_index)
            # Detected "GALACTIC" keyword
            start_time = i * frame_length
            while keyword_index == 0:
                i += 1
                frame = buffer.get()
                keyword_index = porcupine.process(frame)

            end_time = i * frame_length
            list_samples_id = np.arange(start_time, end_time)
            labels = [1 for _ in range(len(list_samples_id))]
            label_samples(list_samples_id, labels)

            # Calculate the duration of the detected keyword interval
            keyword_duration = (end_time - start_time) / sample_rate  # in seconds

            # Make the window size dynamic based on the keyword duration
            window_size_factor = 0.2  # You can adjust this factor based on your needs
            window_size = int(keyword_duration * window_size_factor * sample_rate)

            # Mark a window around the detected keyword as 0
            start_window = max(0, start_time - window_size)
            end_window = min(end_time + window_size, len(audio_data_int16))
            list_samples_id_window = np.arange(start_window, end_window)
            labels_window = [0 for _ in range(len(list_samples_id_window))]
            label_samples(list_samples_id_window, labels_window)

        else:
            list_samples_id = np.arange(i * frame_length, (i + 1) * frame_length)
            labels = [1 for _ in range(len(list_samples_id))]
            label_samples(list_samples_id, labels)

        i += 1

    print('Stop processing')

    
    # Save the list to a file
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)


if __name__ == "__main__": 
    time_measurement = []

    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)

    pygame.mixer.init()
    
    thread_process.start()
    thread_emit.start()