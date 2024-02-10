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

### Solution description (completed)

#### Team

Team name: Enchantix Innovators

Members:

- Anusha Asim - anushaasim21@yahoo.com - College Student at Regent Middle East - Course: BTEC Higher National Diploma in Computing

- Ammar Ahmed Farooqi - ammarahmed9999@yahoo.com - College Student at Regent Middle Ease - Course: BTEC Higher National Diploma in Computing

- Murtaza Mustafa - murtaza.0903@gmail.com - College Student at Heriot Watt University - Course: BSc Hons Computer Science

#### Solution description
The provided solution is designed for the AAICO February 2024 Voice Processing Challenge. The challenge involves processing a given audio file containing voice commands in a streaming simulation scenario. The goal is to detect and label voice commands accurately while simulating real-time constraints.

#### Key Components and Justifications:

1. YAMNet Model:

The solution utilizes the YAMNet (Yet Another Multilabel Network) model from TensorFlow Hub. This pre-trained model is specifically designed for audio event classification and can recognize a diverse set of audio events, making it suitable for detecting specific voice commands in the provided audio file.
Link to YAMNet model: https://www.kaggle.com/models/google/yamnet/frameworks/tensorFlow2/variations/yamnet/versions/1?tfhub-redirect=true

2. Multithreading for Real-time Simulation:

Multithreading is employed to simulate real-time processing constraints. Two threads, thread_process and thread_emit, handle data processing and audio emission simultaneously. This approach ensures that the processing logic adheres to the real-time simulation requirements of the challenge.

3. Data Serialization using Pickle:

The solution uses the pickle module to serialize and save the results in a binary file (results.pkl). 

#### Model Performance 

#### EVALUATION ENVIRONMENT
Based on the code and logic provided by AAICO in the evaluation Google Colab, the team created our own Google Colab Notebook to assess the performance of the model: https://colab.research.google.com/drive/1qi7U9Y1Mz-3Q8L3hKbccKPqISRcnZWNQ?usp=sharing. This was done due to some edits needed in the evaluation code to ensure compatibility with our solution. Our Google Colab file contains the entire solution code, as well as the code assessing model performance.

#### SCORE
Using the logic for the score provided by the challenge, the solution scored 0.0104, indicating the ratio of correctly processed samples within the specified real-time constraints. Achieving a low score is desirable, as it indicates fewer penalties for processing delay, unintentional broadcast of command samples, and loss of communication samples.

#### INDEXING ISSUE and SCORE ADJUSTMENT
During the implementation of the provided scoring code to our solution, an indexing issue occurred, resulting in an IndexError at index 818176, which was addressed by refining the score calculation logic. The updated code ensures compatibility with our solution.

#### Hardware and Environment Recommendations
To run the solution optimally, the following hardware and environment recommendations are suggested:

1. Hardware:
- A machine with sufficient computational resources, especially considering the use of TensorFlow and its underlying neural network model (YAMNet).
Adequate RAM and GPU capabilities for faster processing.

2. Optimal Machine for Testing and Scoring:
- A machine with at least a mid-range GPU, such as NVIDIA GTX 1060 or higher, to accelerate the YAMNet model's computations.
- Recommended RAM: 16 GB or higher.
- A machine running Linux, preferably Ubuntu, for seamless compatibility with TensorFlow and other dependencies.

#### Limitation and Future Improvement 
In real-world scenarios, disruptions or variations in audio input may impact performance. Future iterations could explore handling of such scenarios, incorporating error recovery mechanisms. 

### Submission Deadline

Make sure to submit your solution before February 11th 2024, 11:59pm UAE time.

### Contact

If you have any questions or need clarification on the challenge, feel free to reach out on Discord: https://discord.com/channels/1104007013329014884.

Best of luck!

AAICO team.
