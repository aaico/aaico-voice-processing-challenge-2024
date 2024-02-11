# AAICO February 2024 Voice Processing Challenge

## Welcome

Welcome to the AAICO January 2024 Voice Processing Challenge! This repository contains the necessary resources and information to participate in the challenge. Please read the following guidelines to ensure a smooth and successful participation.

### Challenge Overview

The challenge involves completing the '**aaico_voice_processing_challenge.py**' file. This file simulates the streaming of the '**audio_aaico_challenge.wav**' audio file. Frame by frame, the "emit_data" thread emits the data of the audio file. Each frame consists of 512 samples, with a sample rate of 16000 Hz for the audio file.

The "process_data" thread receives these frames. Your task is to complete the code in this thread to label each received sample and save your label using the provided function "label_samples". A sample should be labeled 0 if it is detected as a command, otherwise 1 (we consider that everything that is not a command should be broadcast).

Once the code is executed, a '**results.pkl**' file will be saved, which is an array containing for each sample:

- The time at which the sample was emitted.
- The label you assigned to the sample.
- The time at which the sample was labelled.

More details on the challenge are provided here: https://docs.google.com/document/d/1Nacv8gT2kfG2wGWXIdKaisStBy2xfGPJIGy27AqqEo4.

You can evaluate your results directly on Colab in which the scoring method is fully explicit: https://colab.research.google.com/drive/1ekMF1UFfr3djseliJleUNpvzfyIJP57G?usp=sharing by uploading the results.pkl file (along with the audio_aaico_challenge.wav file).

### Instructions

To submit your solution, fork the repository, create a branch with the name of your team, push to your branch and then create a pull request to the original repository. Indicate in the Solution description section (below) your team's name, the name and email of each member and a description of your solution.

To have your solution considered, it must be reproducible by the AAICO team.

### Solution description (to complete)

#### Team

Team name: [LoneWolf]

Members:

- [Srinivas Valmeti] - [vsrinivastp@gmail.com]

#### Solution description

The solution makes use of one of the popular open source LLMs, Wav2Vec2ForCTC (refer https://huggingface.co/docs/transformers/model_doc/wav2vec2 for documentation).

The solution involves the following approach/ steps:
1. Since LLMs like Wav2Vec2ForCTC require a sequence of words (more than just one or two words) to capture the context and make inferences, it is important to collect as many frames as possible,befiore inference.
2. Since the given the audio is just about a minute, I decided to capture the all the frames, before carrying out inference and further process
3. Once all the frames are captured, the LLM makes the inferences and captures, the worsd in the audio and their time offsets.
4. The code looks for the words 'GALACTIC' or 'GALECTIC' and capture the start time offset of the word and end time offset of the word (or its next word, if exists), for each instance of the word command occurance (Galactic followed by some other word) as they are required to be blocked in broadcast communication.
5. Based on the captured time_offsets, the labels are encoded and published at a time to results object

The upsides of the solution:
1. It is extremely (almost 100%) accurate in correctly infering the spoken words, identifying the command words and their  corresponding time stamps
2. It is fast in processing long sentences.

The downside of the solution:
 1. The overrun time between individual frame emitted and encoding corresponding labels in results object exceeds teh required threshold by a big factor (for all samples), making the score zero.
2. Such a overrun may not be of real practical use.

### Submission Deadline

Make sure to submit your solution before February 11th 2024, 11:59pm UAE time.

### Contact

If you have any questions or need clarification on the challenge, feel free to reach out on Discord: https://discord.com/channels/1104007013329014884.

Best of luck!

AAICO team.
