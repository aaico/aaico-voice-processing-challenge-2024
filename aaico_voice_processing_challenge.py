import librosa
import numpy as np
import time
import threading
import queue
import pickle
import pvporcupine


porcupine = pvporcupine.create(
  access_key='zhpWuhGi8aOj2ssRk/jqEG1Vj96CjTaXRynJmYlgTQdC9lkPDjINbQ==',
#   keywords=['GALACTIC-BATTERY', 'LACTIC-OXYGEN', 'LACTIC-TEMPERATURE'],
  keyword_paths=['GALACTIC-BATTERY_en_mac_v3_0_0.ppn', 'LACTIC-OXYGEN_en_mac_v3_0_0.ppn', 'LACTIC-TEMPERATURE_en_mac_v3_0_0.ppn'],
)



# porcupine = pvporcupine.create(
#   access_key='QdCMpUJN09KJMeshj/4q/GVzBhMSq4Tzhsqp3ZaDJ86jddRkcEUAdQ==',
# #   keywords=['GALACTIC-BATTERY', 'LACTIC-OXYGEN', 'LACTIC-TEMPERATURE'],
#   keyword_paths=['GALACTIC_en_mac_v3_0_0.ppn', 'LACTIC_en_mac_v3_0_0.ppn'],
# )

# print(porcupine.sample_rate)
# print(porcupine.frame_length)



########### PARAMETERS ###########
# DO NOT MODIFY
# Desired sample rate 16000 Hz
# sample_rate = 16000
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

def process_data():
    i = 0
    start_event.wait()
    print('Start processing')

    while i != number_of_frames:
        frame = buffer.get()
        keyword_index = porcupine.process(frame)
        # 17952
        if keyword_index >=0:
            print(f"Detected keyword at frame {i}")
            print(f"Detected keyword at time {i*frame_length/sample_rate}")
            print("Length of frame: ", len(frame))
            end_time = (i + 1) * frame_length

            if keyword_index == 2:
                print("Detected keyword: GALACTIC-TEMPERATURE")
                start_time = max(0, end_time - 22000)
            elif keyword_index == 0:
                print("Detected keyword: GALACTIC-BATTERY")
                start_time = max(0, end_time - 28060)
            else:
                print("Detected keyword: GALACTIC-OXYGEN")
                start_time = max(0, end_time - 18240)
                # 17952
                
            print(f"Start time: {start_time}")
            print(f"End time: {end_time}")
            
            list_samples_id = np.arange(start_time, end_time)
            labels_keyword = [0 for _ in range(len(list_samples_id))]
            label_samples(list_samples_id, labels_keyword)


        else:
            list_samples_id = np.arange(i * frame_length, (i + 1) * frame_length)
            labels_no_keyword = [1 for _ in range(len(list_samples_id))]
            label_samples(list_samples_id, labels_no_keyword)

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