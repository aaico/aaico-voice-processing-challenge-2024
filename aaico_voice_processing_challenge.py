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
# audio_file = "evaluation_audio_aaico_challenge.wav"
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
    start_event.wait()
    print('Start processing')

    ### TODO: YOUR CODE

    ### Main idea is to divide all the frames into four segments and process each segment separately so that
    ### Wave2VecCTC has enough frames to make meaningful inference and at teh same time the processing is fast enough

    results[1] = np.ones(len(audio_data_int16))
    i = 0
    all_samples = []
    start = 0
    segs = 4

    ## Create a list of 4 elements, each representing the last frame of each segment
    n_frames = list(range(int(number_of_frames / segs), number_of_frames, int(number_of_frames / segs)))

    ## Append the last frame number so that all frames are processed
    if n_frames[-1] != number_of_frames:
        n_frames.append(number_of_frames)
    command_samples = []

    while i != number_of_frames:

        ## collect all the farmes before processing - the LLM inferences correctly only enough frames together
        ## inferencing with one frame at a time will not yield results
        frame = buffer.get()
        all_samples.extend(np.array(frame).astype(np.float64) / 32767)
        #$ We will process each of the segments when the appropriate number of frames are collected as per segments
        if i in n_frames:
            samples = all_samples[start:i * 512]
            ## create range of sampleids covering the entire data
            list_samples_id = np.arange(start, i * 512)
            ## process the data to get best predictions
            input_values = processor(samples, return_tensors="pt", padding="longest",
                                     sampling_rate=sample_rate).input_values  # Batch size 1
            # print(input_values)
            logits = model(input_values).logits
            pred_ids = torch.argmax(logits[0], axis=-1)
            # print(pred_ids)
            ## retrieve word stamps
            outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
            ## compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
            time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate
            # print(time_offset)
            # text = outputs.text.split()
            # print(text)
            ## extract word_offsets from word_stamps
            word_offsets = outputs.word_offsets
            #             print(word_offsets)
            ## find Galactic/ Galectic/ Gollectic word occurence, its start and
            ## end time stamps (end time stamp of its next word, if exists) & collect them iteratively
            ## Initialize a list to store start and end times
            ## Iterate through the word_offsets list to find "GALACTIC" and its next word

            j = 0
            while j < len(word_offsets):
                if word_offsets[j]['word'].startswith('GALAC') or \
                        word_offsets[j]['word'].startswith('GALEC') or \
                        word_offsets[j]['word'].startswith('GOLLEC'):
                    start_time = int(word_offsets[j]['start_offset']) * time_offset + start / sample_rate
                    ## Find the end_offset of the word immediately following "GALACTIC"
                    if j + 1 < len(word_offsets):
                        end_time = int(word_offsets[j + 1]['end_offset']) * time_offset + start / sample_rate
                    else:
                        ## If "GALACTIC" is the last word, set the end_time to its own end_offset
                        end_time = int(word_offsets[j]['end_offset']) * time_offset + start / sample_rate
                    ## Append start and end times to the times_list
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    command_samples.append([start_sample, end_sample])
                j += 1

            ## create labels as per the command_samples collected
            labels = np.ones(len(list_samples_id))
            for k in list_samples_id:
                if any([k >= e[0] and k <= e[1] for e in command_samples]):
                    labels[np.where(list_samples_id == k)] = 0
            # print(command_samples)
            start = i * 512
            n_frames.remove(i)

            # print('labelsum:',sum(labels))
            # publish the labels to results
            label_samples(list_samples_id, labels)
        i += 1

    print(command_samples)

    ###

    # Save the list to a file
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)

    print('Stop processing')


if __name__ == "__main__":
    time_measurement = []

    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)

    thread_process.start()
    thread_emit.start()