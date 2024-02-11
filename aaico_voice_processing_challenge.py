from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer, AutoFeatureExtractor
import torch
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

# load model, processor, feature_extractor and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

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

    ### TODO: YOUR CODE

    ## collect all the frames before processing - the LLM inferences correctly only with all the frames together
    ## inferencing with one frame at a time will not yeirld results
    ## also rescale the frame data to original floating point format as the LLM can process only such data

    frames = []
    while i != number_of_frames:
        frame = buffer.get()
        frames.extend(np.array(frame).astype(np.float64) / 32767)
        i += 1
    # create range of sampleids covering the entire data
    list_samples_id = np.arange(len(audio_data_int16))

    # process the data to get best predictions
    input_values = processor(frames, return_tensors="pt", padding="longest",
                             sampling_rate=sample_rate).input_values  # Batch size 1
    logits = model(input_values).logits
    pred_ids = torch.argmax(logits[0], axis=-1)

    # retrieve word stamps
    outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
    # print(outputs)

    # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
    time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate
    # text = outputs.text.split()

    # extract word_offsets from word_stamps
    word_offsets = outputs.word_offsets

    ## find Galactic/ Galectic word occurence, its start and
    ## end time stamps (end time stamp of its next word, if exists) & collect them iteratively

    # Initialize a list to store start and end times
    command_samples = []
    # Iterate through the word_offsets list to find "GALACTIC" and its next word (if exists)
    i = 0
    while i < len(word_offsets):
        if word_offsets[i]['word'] in ['GALACTIC', 'GALECTIC']:
            start_time = int(word_offsets[i]['start_offset']* time_offset*10**3)
            # Find the end_offset of the word immediately following "GALACTIC"
            if i + 1 < len(word_offsets):
                end_time = int(word_offsets[i + 1]['end_offset']* time_offset*10**3)
            else:
                # If "GALACTIC" is the last word, set the end_time to its own end_offset
                end_time = int(word_offsets[i]['end_offset']* time_offset*10**3)

            # Append start and end times to the times_list
            command_samples.append([start_time, end_time])

        i += 1
    # print(command_samples)

    # create labels as per the command_samples collected
    labels = np.ones(len(list_samples_id))
    for i in range(len(labels)):
        if any([i >= e[0] and i <= e[1] for e in command_samples]):
            labels[i] = 0
    # print('labelsum:',sum(labels))

    # publish the labels to results
    label_samples(list_samples_id, labels)

    print('Stop processing')

    # print('wo:', word_offsets)
    # print('command_samples:', command_samples)
    # print('results:', results)
    ###

    # Save the list to a file
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    time_measurement = []

    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)

    thread_process.start()
    thread_emit.start()