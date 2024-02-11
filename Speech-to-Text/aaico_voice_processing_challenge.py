import librosa
import numpy as np
import time
import threading
import queue
import pickle
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torchaudio
from fuzzywuzzy import fuzz

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

# Load tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

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
    buffered_frames = []
    frame_buffer_size = 50
    print('Start processing')
    while i != number_of_frames:
        frame = buffer.get()
        
        ### TODO: YOUR CODE
        # MODIFY
        buffered_frames.append(frame)

        # Process the buffered frames when the buffer is full or it's the last batch
        if len(buffered_frames) == frame_buffer_size:
            # Process and transcribe the accumulated frames
            accumulated_frames = np.concatenate(buffered_frames)
            waveform = accumulated_frames.astype(np.float32) / 32767
            waveform = np.expand_dims(waveform, axis=0)
            input_values = tokenizer(waveform, return_tensors="pt", padding="longest").input_values

            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = tokenizer.batch_decode(predicted_ids)[0]

            # Check transcription for the command
            threshold = 50
            if fuzz.partial_ratio("GALACTIC", transcription.upper()) > threshold:
                print(transcription)
                label = 0
            else:
                label = 1

            # Label each frame in the current batch
            for frame_i in range(len(buffered_frames)):
                list_samples_id = np.arange((i - len(buffered_frames) + frame_i + 1) * frame_length, 
                                            (i - len(buffered_frames) + frame_i + 2) * frame_length)
                label_samples(list_samples_id, [label] * frame_length)

            # Clear the buffer for the next batch
            buffered_frames = []
        if number_of_frames - i < frame_buffer_size:
            list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
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
    
    thread_process.start()
    thread_emit.start()