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
        list_samples_id = np.arange(i * frame_length, (i + 1) * frame_length)
        time.sleep(frame_length / sample_rate)  # Simulate real time
        frame = audio_data_int16[list_samples_id]
        buffer.put(frame)
        notice_send_samples(list_samples_id)
    print('Stop emitting')


def process_data():
    i = 0
    start_event.wait()
    print('Start processing')

    # playback_start_time = time.time()
    # while i != number_of_frames:
    #     frame = buffer.get()
    #
    #
    #     abs_frame= np.abs(frame)
    #
    #     is_speech = np.max(abs_frame) > 0.05
    #
    #     if is_speech:
    #
    #         is_command = "GALACTIC" in str(frame)
    #
    #         if is_command:
    #             labels= [0] * len(frame)
    #         else:
    #             labels = [1] * len(frame)
    #     else:
    #         labels = [1] * len(frame)
    #
    #         list_samples_id = np.arange(i * frame_length, min((i + 1) * frame_length + len(results[1])))
    #         label_samples(list_samples_id, labels)
    #
    #         i += i
    playback_start_time = time.time()
    detected_galactic_frames = 0
    while i != number_of_frames:
        frame = buffer.get()
        is_command = "GALACTIC" in str(frame)

        if is_command:
            detected_galactic_frames += 1
            i += 1

        playback_end_time = time.time()
        playback_duration = (playback_end_time - playback_start_time) * 1000
        playback_overrun = playback_duration - audio_duration

        print('Stop the process')

        with open('results.pkl', 'wb') as file:
            pickle.dump(results, file)

        ### TODO: YOUR CODE
        # MODIFY
    # command_threshold=5000
    # is_command = np.max(np.abs(frame)) > command_threshold
    # labels=[0 if is_command else 1] * len(frame)
    ## list_samples_id = np.arange(i * frame_length, (i + 1) * frame_length)
    # label_samples(list_samples_id, labels)
    # i += 1

    #  playback_end_time = time.time()
    #  playback_duration = (playback_end_time - playback_start_time) * 1000
    # playback_overrun =  playback_duration - audio_duration
    # print('stop processing')
    # list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
    # labels = [1 for _ in range(len(list_samples_id))]
    ###

    # label_samples(list_samples_id, labels)
    #  i += 1

    # list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
    #   commands = ['galactic temperature', 'galactic oxygen', 'galactic battery']

    #   frame = audio_data_int16[list_samples_id]
    #    recognizer = sr.Recognizer()
    #   audio = sr.AudioData(frame, sample_rate, 1)
    #   try:
    #        text = recognizer.recognize_google(audio)
    #        labels = [0 if _ in commands else 1 for _ in range(len(list_samples_id))]
    #    except:
    #        labels = np.zeros(frame_length, dtype = np.int64)

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