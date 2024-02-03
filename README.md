# AAICO February 2024 Voice Processing Challenge

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

Team name: AIx

Member:

- Ibrahim Patel - ibrahimpatel009@gmail.com

#### Solution description

Provide clear and concise documentation in your code and update the README.md file with any additional information regarding your solution.

