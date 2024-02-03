import torch
import numpy as np
import time
import threading
import queue
import pickle
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio

# Check for MPS (Apple Silicon) support; fall back to CPU otherwise
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Parameters
sample_rate = 16000
frame_length = 512

# Model setup
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)
model.eval()

# Function to preprocess audio
def preprocess_audio(audio_data):
    # Convert numpy array to PyTorch tensor and normalize
    audio_tensor = torch.tensor(audio_data).float()
    max_val = torch.max(torch.abs(audio_tensor))
    audio_normalized = audio_tensor / max_val if max_val > 0 else audio_tensor
    return audio_normalized.numpy()

# Function to decode audio
def decode_audio(audio_data):
    inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription

# Command detection logic
def is_command(transcription):
    for command in ["GALACTIC BATTERY", "GALACTIC OXYGEN", "GALACTIC TEMPERATURE"]:
        if command in transcription.upper():
            return True
    return False

# Streaming and processing setup
audio_file = "audio_aaico_challenge.wav"
audio_data, current_sample_rate = torchaudio.load(audio_file, normalize=True)
audio_data = torchaudio.transforms.Resample(orig_freq=current_sample_rate, new_freq=sample_rate)(audio_data[0]).numpy()
audio_data_int16 = (audio_data * 32767).astype(np.int16)
number_of_frames = len(audio_data_int16) // frame_length
audio_data_int16 = audio_data_int16[:number_of_frames * frame_length]

results = np.zeros(shape=(3, len(audio_data_int16)), dtype=np.int64)
buffer = queue.Queue()
start_event = threading.Event()

# Sample labeling and notice functions
def label_samples(list_samples_id, labels):
    receive_time = time.time_ns()
    results[1][list_samples_id] = labels
    results[2][list_samples_id] = receive_time

def notice_send_samples(list_samples_id):
    send_time = time.time_ns()
    results[0][list_samples_id] = send_time

# Data emission and processing functions
def emit_data():
    time.sleep(.5)
    print('Start emitting')
    start_event.set()
    for i in range(0, number_of_frames):
        list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
        time.sleep(frame_length / sample_rate)
        frame = audio_data_int16[list_samples_id]
        buffer.put(frame)
        notice_send_samples(list_samples_id)
    print('Stop emitting')

def process_data():
    i = 0
    start_event.wait()
    print('Start processing')
    while i != number_of_frames:
        frame = buffer.get()
        audio_chunk = preprocess_audio(frame.astype(np.float32) / 32767)
        transcription = decode_audio(audio_chunk)
        labels = [0 if is_command(trans) else 1 for trans in transcription]
        list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
        label_samples(list_samples_id, labels)
        i += 1
    print('Stop processing')
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)

# Main execution block
if __name__ == "__main__":
    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)
    
    thread_process.start()
    thread_emit.start()
    thread_process.join()
    thread_emit.join()
