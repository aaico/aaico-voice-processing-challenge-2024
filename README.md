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

### Solution description

#### Team

Team name: Data Crusaders

Members:

- Saajida Shajahan - mailsaajida@gmail.com
- Seena MS - contactmeseena508@gmail.com
- Rubaya Mohammed Ali - rubayah.m@gmail.com

#### Requirements

Vosk Installation: Ensure the Vosk library is installed in your Python environment. Vosk can be installed via pip using the following command: pip install vosk

Download a Pre-trained Vosk Model: Select and download an appropriate Vosk pre-trained model for your language and domain from the official Vosk Model page. The model named vosk-model-small-en-us-0.15 is a compact model for English, suitable for quick tests and devices with limited resources. For more accurate recognition, consider larger models that are also available on the Vosk model page.

#### Solution description

The solution here mainly includes two components:

Speech Recognition: The Vosk speech recognition library is utilized to transcribe the audio content of each frame. Vosk is chosen for its offline capabilities and support for real-time processing, making it suitable for applications where internet access is unavailable or latency needs to be minimized.

Command Detection and Labeling: For each frame processed, the transcribed text is checked for the presence of the keyword "galactic," which signifies a command. Frames containing this keyword are labeled as '0', indicating they should not be broadcasted. All other frames are labeled as '1', indicating they are clear for broadcast.


### Submission Deadline

Make sure to submit your solution before February 11th 2024, 11:59pm UAE time.

### Contact

If you have any questions or need clarification on the challenge, feel free to reach out on Discord: https://discord.com/channels/1104007013329014884.

Best of luck!

AAICO team.
