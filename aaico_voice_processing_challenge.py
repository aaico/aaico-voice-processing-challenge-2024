import librosa #audio processing library
import numpy as np #math library
import time #time library
import threading #threading library
import queue #queue library
import pickle #pickle library - it basically serializes the object first before writing it to file

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
    print('Start emitting')
    start_time = time.time()
    start_event.set()
    for i in range(0, number_of_frames):
        list_samples_id = np.arange(i * frame_length, (i + 1) * frame_length)
        frame = audio_data_int16[list_samples_id]
        buffer.put((time.time() - start_time, frame, list_samples_id))  # Fix here
        notice_send_samples(list_samples_id)
    print('Stop emitting')

def detect_command(frame):
    # Implement your command detection logic here based on wake word and specific commands
    # For simplicity, let's assume a basic pattern matching approach
    wake_word_pattern = np.array([0.2, 0.5, 0.7])  # Adjust this based on your audio characteristics
    correlation_wake_word = np.correlate(frame, wake_word_pattern, mode='valid')

    # Set a threshold for wake word detection
    threshold_wake_word = 0.5  # Adjust as needed
    wake_word_detected = np.max(correlation_wake_word) > threshold_wake_word

    if wake_word_detected:
        # Identify the specific command based on the wake word
        command_label = identify_command(frame)

        if command_label is not None:
            return True  # Command detected

    return False  # No valid command detected

def identify_command(frame):
    # Implement logic to identify the specific command after detecting the wake word
    # You may use similar techniques as above, considering the specific characteristics of each command
    battery_pattern = np.array([0.1, 0.4, 0.6])  # Adjust based on characteristics of "BATTERY" command
    oxygen_pattern = np.array([0.3, 0.6, 0.8])   # Adjust based on characteristics of "OXYGEN" command
    temperature_pattern = np.array([0.5, 0.8, 0.9])  # Adjust based on characteristics of "TEMPERATURE" command

    # Perform cross-correlation or another similarity measure for each command
    correlation_battery = np.correlate(frame, battery_pattern, mode='valid')
    correlation_oxygen = np.correlate(frame, oxygen_pattern, mode='valid')
    correlation_temperature = np.correlate(frame, temperature_pattern, mode='valid')

    # Set a threshold for command identification
    threshold_command = 0.5  # Adjust as needed

    # Identify the specific command based on the highest correlation
    if np.max(correlation_battery) > threshold_command:
        return 0  # "BATTERY" command
    elif np.max(correlation_oxygen) > threshold_command:
        return 1  # "OXYGEN" command
    elif np.max(correlation_temperature) > threshold_command:
        return 2  # "TEMPERATURE" command
    else:
        return None  # No valid command detected
    
def process_data():
    i = 0
    start_event.wait()
    print('Start processing')

    while i != number_of_frames:
        start_processing_time = time.time()
        emission_time, frame, list_samples_id = buffer.get()

        # MODIFY: Implement command detection logic
        # Use the provided detect_command function to determine if the frame contains a command
        is_command_frame = detect_command(frame)

        # Update labels based on detection results
        labels = [0 if is_command_frame else 1 for _ in range(len(list_samples_id))]
        label_samples(list_samples_id, labels)
        i += 1

        # Print processing time for each frame
        processing_time = (time.time() - start_processing_time) * 1000  # Convert to milliseconds
        print(f"Processing Time for Frame {i}: {processing_time:.2f} ms")

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