import numpy as np
import time
import threading
import queue
import pickle
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2CTCTokenizer
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import plotly.express as px

# Parameters
sample_rate = 16000
frame_length = 512
audio_file = "audio_aaico_challenge.wav"

# Function to preprocess audio
def preprocess_audio(audio_file, sample_rate, frame_length):
    audio_data, _ = librosa.load(audio_file, sr=sample_rate)
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    number_of_frames = len(audio_data_int16) // frame_length
    audio_data_int16 = audio_data_int16[:number_of_frames * frame_length]

    return audio_data_int16, number_of_frames

# Streaming simulation
results = np.zeros(shape=(3, 0), dtype=np.int64)  # Initialize with 0 columns

# Command samples
command_samples = [
    [142000, 160000],
    [340000, 360000],
    [620000, 635000]
]

nb_command_samples = sum([elem[1] - elem[0] for elem in command_samples])

# Ground truth labels
def create_ground_truth(number_of_frames, command_samples):
    ground_truth = np.ones(number_of_frames)
    for i in range(number_of_frames):
        if any([i >= e[0] and i <= e[1] for e in command_samples]):
            ground_truth[i] = 0
    return ground_truth

# Function to emit data
def emit_data(buffer, number_of_frames, frame_length, sample_rate, start_event):
    time.sleep(.5)
    print('Start emitting')
    start_event.set()
    for i in range(0, number_of_frames):
        list_samples_id = np.arange(i * frame_length, (i + 1) * frame_length)
        time.sleep(frame_length / sample_rate)  # Simulate real-time
        frame = audio_data_int16[list_samples_id]
        buffer.put((frame, list_samples_id))
        notice_send_samples(list_samples_id)
    print('Stop emitting')

# Function to process data
def process_data(buffer, number_of_frames):
    i = 0
    start_event.wait()
    print('Start processing')

    while i != number_of_frames:
        frame, list_samples_id = buffer.get()

        # Convert features to predictions using the trained model
        inputs = tokenizer(frame.tolist(), return_tensors="pt", padding="max_length", truncation=True, max_length=frame_length, stride=frame_length, sampling_rate=sample_rate)
        outputs = model(**inputs)
        predictions = outputs.logits

        # Assuming binary classification
        labels = (predictions > 0).astype(np.int)

        label_samples(list_samples_id, labels)
        i += 1

    print('Stop processing')

    # Save the list to a file
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)

    # Display information about command samples
    print(f"Number of command samples: {nb_command_samples}")

    # Evaluate the performance
    overrun_times_ms = (results[2] - results[0]) / 1e6
    labels = results[1]

    assert np.all(np.diff(results[2]) >= 0)  # Labelling has been done sequentially
    assert np.all(overrun_times_ms <= 50)  # Processing took less than 50 ms for each sample

    slow_sample_labelling_thres = 20
    command_ratio = nb_command_samples / len(audio_data_int16)
    communication_ratio = 1 - nb_command_samples / len(audio_data_int16)

    score = len(audio_data_int16)

    for i in range(len(audio_data_int16)):
        if overrun_times_ms[i] >= slow_sample_labelling_thres:
            score -= 1
        else:
            if ground_truth[i] == 0 and labels[i] != 0:  # unintentional broadcast
                score -= int(1 / command_ratio)
            if ground_truth[i] == 1 and labels[i] != 1:  # lost communication
                score -= int(1 / communication_ratio)

    print(f'Score: {score / len(audio_data_int16)}')

if __name__ == "__main__":
    # Invoke the preprocess_audio function
    audio_data_int16, number_of_frames = preprocess_audio(audio_file, sample_rate, frame_length)

    # Display audio
    display(Audio(audio_data_int16[620000: 627000], rate=16000))

    # Plot audio
    fig = px.scatter(audio_data_int16, title="Input audio")
    for elem in command_samples:
        fig.add_vline(x=elem[0], line_color="red")
        fig.add_vline(x=elem[1], line_color="red")
    fig.show()

    # Start the threads
    time_measurement = []

    # Create ground truth
    ground_truth = create_ground_truth(number_of_frames, command_samples)

    # Create threads
    buffer = queue.Queue()
    start_event = threading.Event()

    thread_emit = threading.Thread(target=emit_data, args=(buffer, number_of_frames, frame_length, sample_rate, start_event))
    thread_process = threading.Thread(target=process_data, args=(buffer, number_of_frames))

    thread_emit.start()
    thread_process.start()
