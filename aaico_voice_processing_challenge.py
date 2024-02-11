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


# initialize variables
current_word = ""
current_word_frames = 0
galactic_word = False
last_frame_label = 1

def process_current_word(word, frame_index):
    """
    Process the completed word and label the corresponding frames in the `results` array.

    Parameters:
    word (str): The completed word to be processed.
    frame_index (int): The index of the first frame in the word.
    """
    if word == "galactic":
        for i in range(frame_index, frame_index + frame_length):
            results[1][i] = 1
    else:
        for i in range(frame_index, frame_index + frame_length):
            results[1][i] = 0

def process_data():
    """
    Process the audio data in real-time and label the corresponding frames with detected words.
    """
    i = 0
    current_word = ""
    current_word_frames = 0
    start_event.wait()
    print('Start processing')
    while i != number_of_frames:
        frame = buffer.get()

        # Convert the audio data to characters
        try:
            current_word += frame.tobytes().decode('utf-8')
        except UnicodeDecodeError:
            # Handle decoding errors by replacing invalid bytes with a placeholder character
            current_word += frame.tobytes().decode('utf-8', errors='replace')

        current_word_frames += frame_length

        # Check if the word is complete
        if current_word_frames >= frame_length:
            # Process the current word
            process_current_word(current_word, i * frame_length)

            # Reset variables for the next word
            current_word = ""
            current_word_frames = 0

        i += 1

    print('Stop processing')

    # Process any remaining partial word
    if current_word_frames > 0:
        process_current_word(current_word, i * frame_length)

    # Save the results to a file
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)




if __name__ == "__main__": 
    time_measurement = []

    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)
    
    thread_process.start()
    thread_emit.start()
