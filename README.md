<h1>AAICO February 2024 Voice Processing Challenge</h1>

<h2>Welcome</h2>

Welcome to the AAICO January 2024 Voice Processing Challenge! This repository contains the necessary resources and information to participate in the challenge. Please read the following guidelines to ensure a smooth and successful participation.

<h2>Challenge Overview</h2>

The challenge involves completing the 'aaico_voice_processing_challenge.py' file. This file simulates the streaming of the 'audio_aaico_challenge.wav' audio file. Frame by frame, the "emit_data" thread emits the data of the audio file. Each frame consists of 512 samples, with a sample rate of 16000 Hz for the audio file.

The "process_data" thread receives these frames. Your task is to complete the code in this thread to label each received sample and save your label using the provided function "label_samples". A sample should be labeled 0 if it is detected as a command, otherwise 1 (we consider that everything that is not a command should be broadcast).

Once the code is executed, a 'results.pkl' file will be saved, which is an array containing for each sample:

The time at which the sample was emitted.
The label you assigned to the sample.
The time at which the sample was labelled.
More details on the challenge are provided here: https://docs.google.com/document/d/1Nacv8gT2kfG2wGWXIdKaisStBy2xfGPJIGy27AqqEo4.

You can evaluate your results directly on Colab in which the scoring method is fully explicit: https://colab.research.google.com/drive/1ekMF1UFfr3djseliJleUNpvzfyIJP57G?usp=sharing by uploading the results.pkl file (along with the audio_aaico_challenge.wav file).

<h2>Solution description</h2>

<h4>Team name: Encrypters</h4>

<h4>Members:</h4>

Taqhveem Abbas - taqhveem@gmail.com <br>
Omar Abdullah - amooreee34@gmail.com <br>
Syedul Azam - syedulazam6000@gmail.com

<h2>Documentaion</h2>

This documentation provides a step-by-step guide on how to use and understand the provided code for the AAICO Voice Processing Challenge. The challenge involves processing a streaming audio file, labeling each sample, and saving the results. The code is divided into three main parts: audio streaming simulation, sample labeling, and training a classification model.

<h2>1. Prerequisites</h2>

Before using the code, ensure you have the required dependencies installed. You can install them using the following commands:

`pip install numpy librosa transformers matplotlib ipython plotly pandas keras`

Additionally, make sure to have a working Python environment (Python 3.6 or higher).

<h2>2. Code Overview</h2>
   
<h3> 2.1 aaico_voice_processing_challenge.py</h3>

This script orchestrates the streaming simulation and sample labeling process. It involves two threads: one for emitting data (emit_data) and another for processing data (process_data). Follow the steps below:

<h4>2.1.1. Audio Preprocessing</h4>

The preprocess_audio function reads the audio file, resamples it, and prepares it for streaming.

<h4>2.1.2. Streaming Simulation</h4>

Command samples and ground truth labels are defined.
The create_ground_truth function generates ground truth labels based on command samples.
The emit_data function simulates real-time audio streaming and emits data frames into a buffer queue.

<h4>2.1.3. Data Processing</h4>

The process_data function processes data frames using a pre-trained model (model), labels each sample, and saves results in results.pkl.

<h4>2.1.4. Evaluation</h4>

The script evaluates the performance based on predefined criteria, including processing time and accuracy.
The overrun_times_ms variable measures processing time, and the labels variable stores labeled samples.

<h4>2.1.5. Execution</h4>

Execute the script to run the streaming simulation and sample labeling process.

<h3>2.2 dataset_creation_code.py</h3>

This script generates labeled samples to be used for training the classification model. Follow these steps:

<h4>2.2.1. Audio Reading and Resampling</h4>

Reads the audio file and resamples it to the desired sample rate.

<h4>2.2.2. Streaming and Labeling Simulation</h4>

Simulates real-time processing and extracts features from audio frames.
Labeled samples are saved in label_samples.pkl.

<h4>2.2.3. Execution</h4>

Execute the script to generate labeled samples.

<h3>2.3 train_model.py</h3>

This script trains a classification model using labeled samples. Follow these steps:

<h4>2.3.1. Loading Labeled Data</h4>

Loads labeled samples from label_samples.pkl.

<h4>2.3.2. Model Architecture</h4>

Defines a neural network model architecture using Keras.

<h4>2.3.3. Training</h4>

Trains the model using a combination of original and Wav2Vec2 features.

<h4>2.3.4. Execution</h4>

Execute the script to train the model and save the trained model in final_model.h5.

<h2>3. Execution Steps</h2>
   
Follow these steps to use the provided code:

**Preprocess Audio:**

Execute aaico_voice_processing_challenge.py to preprocess the audio and start the streaming simulation.

**Generate Labeled Samples:**

Execute dataset_creation_code.py to generate labeled samples.

**Train Classification Model:**

Execute train_model.py to train the classification model using the labeled samples.

**Review Results:**

Check the console for performance metrics.
Results, including timestamps, labels, and processing times, are saved in results.pkl.
Trained model, training history, and labeled samples are saved in final_model.h5, training_history.pkl, and label_samples.pkl, respectively.

<h2>4. Troubleshooting</h2>
   
If you encounter any issues during execution, check for error messages and ensure that all dependencies are installed. Verify that the input audio file (audio_aaico_challenge.wav) is available.
