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

Team name: Crpyt

Members:

- Raj Dinesh Jagasia - rajdj26103@gmail.com - Undergraduate Computer Science Student
- Kevin Joseph Dinesh - kj2011@hw.ac.uk - Undergraduate Computer Science Student
- Abhinav Kottayil - akk2002@hw.ac.uk - Undergraduate Computer Science Student

#### Solution description

The provided solution describes a deep learning approach tailored to process audio signals within a high-stakes firefighting environment, where voice commands control a modern firefighting suit. The core of this solution is a custom Convolutional Neural Network (CustomCNN), designed to differentiate between command signals and general communication. This network features a carefully structured architecture that starts with a convolution layer to capture spatial hierarchies in the input spectrogram data, followed by additional convolution layers that expand the channel depth for more complex feature extraction.

Incorporation of depthwise separable convolutions, aiming to enhance model efficiency by reducing computational costs while maintaining performance. The model concludes with a global average pooling layer that reduces spatial dimensions to a singular vector, feeding into a fully connected layer for binary classification. Each layer is accompanied by batch normalization and ReLU activation functions to stabilize learning and introduce non-linearity, respectively.

Data preprocessing converts audio frames into spectrogram representations, standardizing them for consistent model input. Coupled with batch processing aims to process audio frames in small batches, allowing for faster throughput and reduced processing times per sample. This method ensures that the model meets real-time processing constraints, crucial for operational viability in emergency situations.

Achieved a score of 0.05497689026818906 when evaluated against the provided Collab notebook.

### Submission Deadline

Make sure to submit your solution before February 11th 2024, 11:59pm UAE time.

### Contact

If you have any questions or need clarification on the challenge, feel free to reach out on Discord: https://discord.com/channels/1104007013329014884.

Best of luck!

AAICO team.
