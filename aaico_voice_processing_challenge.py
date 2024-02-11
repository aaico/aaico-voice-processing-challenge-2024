import librosa
import numpy as np
import time
import threading
import queue
import pickle


from fastdtw import fastdtw
template1 = np.load('template1.npy')
template2 = np.load('template2.npy')
template3 = np.load('template3.npy')


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

def is_galactic(frame, template, threshold):
    distance,_ = fastdtw(frame, template)
    if distance < threshold:
        return 1
    else:
        return 0

def process_data():
    i = 0
    start_event.wait()
    print('Start processing')
    working_set = []
    till_next = 0
    list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
    labels = [0 for _ in range(len(list_samples_id))]
    while i != number_of_frames:
        frame = buffer.get()
        
        ### TODO: YOUR CODE
        # MODIFY
        if i>26:
            del working_set[:2]
            ans1 = is_galactic(working_set,template1,100000) 
            ans2 = is_galactic(working_set,template2,100000) 
            ans3 = is_galactic(working_set,template3,100000) 

            ans = ans1 | ans2 | ans3
            
            if ans>0:
                # print(i-25)
                till_next = 35
            if till_next >0:
                list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
                labels = [1 for _ in range(len(list_samples_id))]
            else:
                list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
                labels = [0 for _ in range(len(list_samples_id))]
            till_next-=1
        label_samples(list_samples_id, labels)
        for j in range(0,512,256):
            working_set.append(max(frame[j:]))
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