import librosa
import numpy as np
import time
import threading
import queue
import pickle

########### PARAMETERS ###########
# DO NOT MODIFY
# Desired sample rate 16000 Hz
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
 
#importing necessary libraries   
import json
from vosk import Model, KaldiRecognizer
import os

#initializing vosk model
current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, "vosk-model-small-en-us-0.15")
model = Model(model_path)

#creating our own grammar for efficiency
rec = KaldiRecognizer(model, sample_rate, '["ga", "lac", "tic", "[unk]"]')
rec.SetWords(False)


def process_data():
    
    i = 0
    # boolean for checking if message starts with ga
    ga = False
    # start index for batching frames
    start_i = -1;
    # galactic frames
    galactic = -1
    start_event.wait()
    print('Start processing')
    while i != number_of_frames:
        
        frame = buffer.get()

        #check if word is complete
        if rec.AcceptWaveform(b''.join(frame)):
            word = json.loads(rec.Result())['text']
            list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
            #if word ends in galactic then label 0 else 1
            if word.endswith("ga lac tic"):
                labels = [0 for _ in range(len(list_samples_id))]
                label_samples(list_samples_id, labels)
            else:
                labels = [1 for _ in range(len(list_samples_id))]
                label_samples(list_samples_id, labels)
        else:
            word = json.loads(rec.PartialResult())['partial']
            if galactic <= -1:
                #set start index for batch labelling
                if not ga and word.endswith("ga"):
                    ga = True
                    start_i = i
                #if there is a possibility of the next frame being galactic, dont do anything
                elif ga and word.endswith("ga") or word.endswith("ga lac"):
                    pass
                #if word is galactic, batch label frames as 0 from start_i and then reset values
                elif ga and word.endswith("ga lac tic"):
                                 
                    list_samples_id = np.arange(start_i*frame_length, (i+1)*frame_length)
                    labels = [0 for _ in range(len(list_samples_id))]
                    label_samples(list_samples_id, labels)
                    
                    ga = False
                    galactic = 9
                #if word is not galactic, batch label frames as 1 from start_i and then reset values
                elif ga and (word=="" or word.endswith("[unk]") or word.endswith("lac") or word.endswith("tic")):
                    list_samples_id = np.arange(start_i*frame_length, (i+1)*frame_length)
                    labels = [1 for _ in range(len(list_samples_id))]
                    label_samples(list_samples_id, labels)
                        
                    ga = False
                #if non of the above are true, just label this frame as 1
                else:
                    list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
                    labels = [1 for _ in range(len(list_samples_id))]
                    label_samples(list_samples_id, labels)
                    ga = False
            else:
                #if it is galactic, label until the frame ends in galactic and then the next 5120 frames as 0
                list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
                labels = [0 for _ in range(len(list_samples_id))]
                label_samples(list_samples_id, labels)
                if not word.endswith("ga lac tic"):
                    galactic -= 1
        i+=1   
     
        
    print('Stop processing')
    
    #Save the list to a file
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)


if __name__ == "__main__": 
    time_measurement = []

    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)
    
    thread_process.start()
    thread_emit.start()
    