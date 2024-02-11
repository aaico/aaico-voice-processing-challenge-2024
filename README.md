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
Real-time Speech Recognition with Vosk and Python

This code demonstrates real-time speech recognition using the Vosk speech recognition library in Python. The code uses a threading model to simulate real-time processing and emitting of audio frames.

##Requirements

Python 3.6 or higher
Vosk speech recognition library
A pre-trained Vosk model (vosk-model-small-en-us-0.15)

##Setup

Install the Vosk library by following the instructions on the Vosk website (https://alphacephei.com/vosk/install).
Download a pre-trained Vosk model from the Vosk website (https://alphacephei.com/vosk/models) and extract it to a directory.
Modify the model_path variable in the code to point to the directory containing the Vosk model.

##Usage

Run the code in a Python environment.
The code will start emitting audio frames to the Vosk recognizer.
The Vosk recognizer will process the audio frames in real-time and output the recognized speech.
The recognized speech will be labeled as either 0 or 1 based on the recognized words.
The labeled speech and timing information will be stored in the results array.
The results array will be saved to a pickle file (results.pkl) at the end of processing.
#### Team

Team name: [AI]

Members:

- [Hanin Atwany] - [hanin.atwany@mbzuai.ac.ae]
- [Member Name] - [Member email]
- [Member Name] - [Member email]

#### Solution description

Provide clear and concise documentation in your code and update the README.md file with any additional information regarding your solution.

### Submission Deadline

Make sure to submit your solution before February 11th 2024, 11:59pm UAE time.

### Contact

If you have any questions or need clarification on the challenge, feel free to reach out on Discord: https://discord.com/channels/1104007013329014884.

Best of luck!

AAICO team.
