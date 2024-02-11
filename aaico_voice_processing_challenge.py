import librosa
import numpy as np
import time
import threading
import queue
import pickle
import whisper
import torch
import audioop
import speech_recognition as sr
import math
import collections

class WaitTimeoutError(Exception):
    pass

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

# Loading audio model
audio_model = whisper.load_model("small") # use "medium" or "large.in" for higher accurancy
audio_list = []
energy_level = [True]
stream_live = [False]
data_queue = queue.Queue()
audio_data = queue.Queue()
wake_word = ['GALACTIC']
commands = ['BATTERY', 'TEMPERATURE', 'OXYGEN']
control_commands = [i+' '+j for i in wake_word for j in commands]

def trans_aud():
    while audio_list:
        sample = audio_list[0]
        splitted_aud = sample[0]
        start = sample[1]
        end = sample[2]
        audio_list.remove(sample)

        np_data = np.frombuffer(splitted_aud, np.int16).flatten().astype(np.float32) / 32768
        result = audio_model.transcribe(np_data, fp16=torch.cuda.is_available())
        stt = result['text'].strip().replace('.','')
        if stt.upper() in control_commands:
            for i in range(start, end + 1):
                list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
                labels = [0 for _ in range(len(list_samples_id))]
                label_samples(list_samples_id, labels)
        else:
            for i in range(start, number_of_frames):
                list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
                labels = [1 for _ in range(len(list_samples_id))]
                label_samples(list_samples_id, labels)
    
def listen(timeout=None):
	SAMPLE_RATE = 16000
	pause_threshold = 0.0125
	phrase_threshold = 0.0125
	non_speaking_duration = 0.0125
	energy_threshold = 300
	# may or may not required
	dynamic_energy_threshold = True
	dynamic_energy_adjustment_damping = 0.15
	dynamic_energy_ratio = 1.5

	seconds_per_buffer = float(frame_length) / SAMPLE_RATE
	pause_buffer_count = int(math.ceil(pause_threshold / seconds_per_buffer))  # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
	phrase_buffer_count = int(math.ceil(phrase_threshold / seconds_per_buffer))  # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
	non_speaking_buffer_count = int(math.ceil(non_speaking_duration / seconds_per_buffer))  # maximum number of buffers of non-speaking audio to retain before and after a phrase

	# read audio input for phrases until there is a phrase that is long enough
	elapsed_time = 0  # number of seconds of audio read
	buffer = b""  # an empty buffer means that the stream has ended and there is no data left to read
	while True:
		frames = collections.deque()

		# store audio input until the phrase starts
		while True:
			# handle waiting too long for phrase by raising an exception
			elapsed_time += seconds_per_buffer
			if timeout and elapsed_time > timeout:
				raise WaitTimeoutError("listening timed out while waiting for phrase to start")

			buffer = data_queue.get()
			if len(buffer) == 0: break  # reached end of the stream
			frames.append(buffer)
			if len(frames) > non_speaking_buffer_count:  # ensure we only keep the needed amount of non-speaking buffers
				frames.popleft()

			# detect whether speaking has started on audio input
			energy = audioop.rms(buffer, 2)  # energy of the audio signal
			if energy > energy_threshold: break

			# dynamically adjust the energy threshold using asymmetric weighted average
			if dynamic_energy_threshold:
				damping = dynamic_energy_adjustment_damping ** seconds_per_buffer  # account for different chunk sizes and rates
				target_energy = energy * dynamic_energy_ratio
				energy_threshold = energy_threshold * damping + target_energy * (1 - damping)

		# read audio input until the phrase ends
		pause_count, phrase_count = 0, 0
		while True:
			# handle phrase being too long by cutting off the audio
			elapsed_time += seconds_per_buffer

			buffer = data_queue.get()
			if len(buffer) == 0: break  # reached end of the stream
			frames.append(buffer)
			phrase_count += 1

			# check if speaking has stopped for longer than the pause threshold on the audio input
			energy = audioop.rms(buffer, 2)  # unit energy of the audio signal within the buffer
			if energy > energy_threshold:
				pause_count = 0
			else:
				pause_count += 1
			if pause_count > pause_buffer_count:  # end of the phrase
				break

		# check how long the detected phrase is, and retry listening if the phrase is too short
		phrase_count -= pause_count  # exclude the buffers for the pause before the phrase
		if phrase_count >= phrase_buffer_count or len(buffer) == 0: break  # phrase is long enough or we've reached the end of the stream, so stop listening

	# obtain frame data
	for _ in range(pause_count - non_speaking_buffer_count): frames.pop()  # remove extra non-speaking frames at the end
	frame_data = b"".join(frames)

	return sr.AudioData(frame_data, SAMPLE_RATE, 2)

def listen_in_background(callback,):
	running = [True]

	def threaded_listen():
		# with source as s:
			while running[0]:
				try:  # listen for 1 second, then check again if the stop function has been called
					audio = listen(.25)
				except WaitTimeoutError:  # listening timed out, just try again
					pass
				else:
					if running[0]: callback(audio)

	def stopper(wait_for_stop=True):
		running[0] = False
		if wait_for_stop:
			listener_thread.join()  # block until the background thread is done, which can take around 1 second

	listener_thread = threading.Thread(target=threaded_listen)
	listener_thread.daemon = True
	listener_thread.start()
	return stopper

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

    transcribe_once_intiated = False
    splitted_aud = bytes()
    phrase_time = None
    phrase_timeout = 0.0125
    phrase_complete = False
    start_index = 0
    break_index = 0
    def record_callback(audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        audio_data.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    listen_in_background(
        record_callback,
    )

    while i != number_of_frames:
        frame = buffer.get()
        
        ### TODO: YOUR CODE
        if frame is not None: data_queue.put(frame)

        try:
            now = time.time()

            # Pull raw recorded audio from the queue.
            if not audio_data.empty():

                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > phrase_timeout:
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not audio_data.empty():
                    frame = audio_data.get()
                    splitted_aud += frame

            if phrase_complete:

                if break_index:
                    start_index = break_index + 1
                break_index = i 
                global audio_list
                audio_list += [(splitted_aud, start_index, break_index)]
                if not transcribe_once_intiated:
                    thread_trans_aud = threading.Thread(target=trans_aud, args=([]))
                    thread_trans_aud.start()
                if not thread_trans_aud.is_alive():
                    thread_trans_aud = threading.Thread(target=trans_aud, args=([]))
                    thread_trans_aud.start()
                transcribe_once_intiated = True
                phrase_complete = False
                splitted_aud = bytes()

        except Exception as error:
            print('Exception is ',error)
            break

        # MODIFY
        list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
        labels = [1 for _ in range(len(list_samples_id))]
        ###

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