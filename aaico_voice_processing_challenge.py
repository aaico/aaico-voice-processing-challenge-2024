import librosa
import numpy as np
import time
import threading
import queue
import pickle
from vosk import Model, KaldiRecognizer
import json
import os


########### PARAMETERS ###########
# DO NOT MODIFY
# Desired sample rate 16000 Hz
sample_rate = 16000
# Frame length
frame_length = 512

model_path = "vosk-model-small-en-us-0.15"
if not os.path.exists(model_path):
    print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)
vosk_model = Model(model_path)

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
    recognizer = KaldiRecognizer(vosk_model, sample_rate)
    i = 0
    start_event.wait()
    print('Start processing')
    
    while i != number_of_frames:
        frame = buffer.get()
        frame_bytes = np.frombuffer(frame, np.int16).tobytes()
        
        if recognizer.AcceptWaveform(frame_bytes):
            result = json.loads(recognizer.Result())
            text = result.get('text', '').lower()
            
            # Check if the recognized text is a command
            is_command = "galactic" in text 
            
            print(f"Frame {i}: {'Command detected' if is_command else 'Communication'} - {text}")
        else:
            is_command = False  # If nothing is recognized, consider it communication
        
        list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
        labels = [0 if is_command else 1 for _ in range(len(list_samples_id))]
        
        label_samples(list_samples_id, labels)
        
        i += 1
    
    print('Stop processing')
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)

if __name__ == "__main__": 
    time_measurement = []

    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)
    
    thread_process.start()
    thread_emit.start()
