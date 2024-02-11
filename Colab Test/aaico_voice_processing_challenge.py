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

#############MYCODE: START############
#imports
from collections import deque
import os
import torch.nn as nn
import torch
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
# from model import Net
from model import Net
import pandas as pd
#configs
DEBUG_MODE=True

MAXLEN = 13000
MELSPEC_MODEL_PATH = "./melspectrogram.onnx"
EMBEDDING_MODEL_PATH = "./embedding_model.onnx"
DEVICE='cuda'
MODEL_INPUT_SHAPE = 96
MODEL_LAYER_DIM = 128
MODEL_N_BLOCKS = 32
MODEL_N_CLASSES = 1
MODEL_WEIGHTS_PATH = "./28_model.pt"
PRED_THRESHOLD = 0.27
PRIMING_AUDIO="./0.wav"
#INIT
MEMORY = deque(maxlen=MAXLEN)
prime_audio = librosa.load(PRIMING_AUDIO, sr=sample_rate)[0]
prime_audio = prime_audio[:MAXLEN]
prime_audio = (prime_audio * 32767).astype(np.int16)
while (len(MEMORY)<MAXLEN):
    MEMORY.extend(prime_audio)


sessionOptions = ort.SessionOptions()
sessionOptions.inter_op_num_threads = 1
sessionOptions.intra_op_num_threads = 1
device='gpu'
melspec_model = ort.InferenceSession(MELSPEC_MODEL_PATH, sess_options=sessionOptions,
                                                      providers=["CUDAExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"])
onnx_execution_provider = melspec_model.get_providers()[0]
melspec_model_predict = lambda x: melspec_model.run(None, {'input': x})
embedding_model = ort.InferenceSession(EMBEDDING_MODEL_PATH, sess_options=sessionOptions,
                                            providers=["CUDAExecutionProvider"] if device == "gpu"
                                            else ["CPUExecutionProvider"])
embedding_model_predict = lambda x: embedding_model.run(None, {'input_1': x})[0].squeeze()

MODEL = Net(MODEL_INPUT_SHAPE, MODEL_LAYER_DIM, MODEL_N_BLOCKS, MODEL_N_CLASSES)
MODEL.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
MODEL = MODEL.to(DEVICE)
MODEL.eval()

#functions
def load_process(sample):

    if len(sample) < MAXLEN:
        sample_extract = np.pad(sample, (MAXLEN - len(sample), 0), mode='constant')
        # sample_extract = np.pad(sample, (0, MAXLEN - len(sample)), mode='constant')
    else:
        # Generate a random starting index to extract a sequence of length MAXLEN
        start_index = np.random.randint(0, len(sample) - MAXLEN + 1)
        # print(start_index,start_index+MAXLEN)
        sample_extract=  sample[start_index:start_index + MAXLEN]
    # sample_extract = np.pad(sample_extract, (13000 - len(sample_extract),0), mode='constant')
    mel_bins = 32
    x = sample_extract.reshape(1,-1)
    n_frames = int(np.ceil(x.shape[1]/160-3))
    # print(x.shape,n_frames)
    # melspecs = np.empty((x.shape[0], n_frames, mel_bins), dtype=np.float32)
    x = x.astype(np.float32) 
    outputs = melspec_model_predict(x)
    # print((time.time()-start)*1e3)
    # print(outputs[0].shape)
    spec = np.squeeze(outputs[0])/10+2

    x= spec.reshape(1,*spec.shape)
    embedding_dim = 96  # fixed by embedding model
    # print(x.shape)
    n_frames = (x.shape[1] - 76)//8 + 1
    # print(n_frames)
    melspec = x[0]
    # print(melspec.shape)
    window_size = 76
    batch = []
    for i in range(0, melspec.shape[0], 8):
        window = melspec[i:i+window_size]
        if window.shape[0] == window_size:  # ignore windows that are too short (truncates end of clip)
            batch.append(window)
    batch = np.array(batch).astype(np.float32)
    # print(batch.shape)
    # start_test = time.time()
    result = embedding_model_predict(batch.reshape(*batch.shape,1)).reshape(1,-1)
    # print((time.time()-start_test)/1000000)
    embds = torch.tensor(result,dtype=torch.float32)
    return embds
def getpreds(sample):
    try:
        with torch.no_grad():
            start_time = time.time()
            mels = load_process(sample).to(DEVICE)
            step1_time = (time.time()-start_time)*1000
            start_time = time.time()
            preds = MODEL(mels).detach().flatten().cpu().numpy().item()
            step2_time = (time.time()-start_time)*1000
            # print(int(step1_time),int(step2_time))
    except:
        preds = 0
    preds = 1- preds
    
    return preds
print("warming up gpu ....")

time_taken = 1000
for _ in tqdm(range(1024),total=1024,unit="iter"):
    start_time = time.time()
    __ = getpreds(np.array(MEMORY))
    time_taken = (time.time()-start_time)*1000
    # print("latency",time_taken)
    if _%100 == 0:
        tqdm.write(f"Latency: {time_taken:.2f} ms")

print('done....')
print('done....')

if DEBUG_MODE:
    DEBUG_PREDS = []
    DEBUG_PREDSCORES = []
    DEBUG_SIGNALS = []
    DEBUG_SAMPLE_IDS = []
    DEBUG_TIMES = []


#############MYCODE: END########################


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
    if DEBUG_MODE:
        DEBUG_TIMES.extend((results[2][list_samples_id] - results[0][list_samples_id])/1E6)

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
        
        ### TODO: YOUR CODE
        # MODIFY
        #store in memory
        list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
        MEMORY.extend(frame)
        memory  = np.array(MEMORY)
        predscore = getpreds(memory)
        labels = [0 if predscore<PRED_THRESHOLD else 1 for _ in list_samples_id]
        if DEBUG_MODE:
            DEBUG_PREDS.extend(labels)
            DEBUG_PREDSCORES.extend([predscore for _ in list_samples_id])
            DEBUG_SIGNALS.extend(frame)
            DEBUG_SAMPLE_IDS.extend(list_samples_id)
        

        
        
        ###

        label_samples(list_samples_id, labels)
        i += 1
    print('Stop processing')
    # Save the list to a file
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)
    
    if DEBUG_MODE:
        dfdebug = pd.DataFrame({
            'sample_id': DEBUG_SAMPLE_IDS,
            'label': DEBUG_PREDS,
            'predscore': DEBUG_PREDSCORES,
            'time': DEBUG_TIMES,
            'signal': DEBUG_SIGNALS
        })
        os.makedirs("logdir",exist_ok=True)
        dfdebug.to_csv('logdir/debug.csv')


if __name__ == "__main__": 
    time_measurement = []

    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)
    
    thread_process.start()
    thread_emit.start()