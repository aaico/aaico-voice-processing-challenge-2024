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

Team name: Ctrl X

Members:

- Mohammed Riaz - mohd.riaz.2002@gmail.com - BSc Computer Science Student at the University of Wollongong in Dubai
- Mohammed Ejazzur Rahman - mohammedejazzur@gmail.com - BSc Computer Science Student at the University of Wollongong in Dubai
- Hadiyya Mattummathodi - hadiyyasakkir@gmail.com - BSc Computer Science Student at the University of Wollongong in Dubai

#### Solution description

The solution makes use of Vosk (https://alphacephei.com/vosk/) which uses KaldiRecognizer.
We have used the "vosk-model-small-en-us-0.15" model which is a lightweight version designed for small devices such as the Raspberry PI.

##### The solution follows the following steps:

1. Updating the grammar of KaldiRecognizer to "ga", "lac", "tic" for detecting "galactic" and "[unk]" for detecting unknown words. This is done for efficiency and for simpler detection.
2. At each iteration we check the output received from KaldiRecognizer's PartialResult. We check if the word ends with "ga". If so we set start_i to the current i value. Then we check for "ga la" and if it is detected we let it move on to the next iteration without labelling. Lastly we check for "ga lac tic" and if it is found we label all frames from start_i to the current i value as 0. After detecting "ga lac tic" we also label all consecutive frames that end with "ga lac tic" as 0. We also label 10 frames after that as 0 to label the second part of their command after they say galactic. If "ga lac tic" is not found at all we just label them as 1.
3. If the word does not end with "ga" in the current iteration we just label it as 1.


##### Advantages:

Our solution is really fast and can run on small devices such as the Raspberry PI as it is very efficient and lightweight.

##### Disadvantages:

It incorrectly labels a few samples (especially before it detects "ga"). The solution also expects the user to say their command within 0.32 seconds after galactic.

##### Score

As some of the samples are processed above 50ms we had to comment out "assert np.all(overrun_times_ms <= 50)" for it to actually score our solution. Our solution received a score of 0.603/1. Majority of our scores were lost because of accuracy, however to satisfy the 20ms time constraint we decided to give up a bit of accuracy in order to receive a higher score.

Our solution uses batch labelling for some of the samples which is why it does not satisfy the 50ms rule. However, 94% of our samples are processed under 20ms out of which 97% of the samples are labelled correctly.

##### Requirements

Along with librosa and numpy, you need to install vosk for our solution to work. Model files are included with the solution. Our solution also uses json and os but they are included with python.

### Submission Deadline

Make sure to submit your solution before February 11th 2024, 11:59pm UAE time.

### Contact

If you have any questions or need clarification on the challenge, feel free to reach out on Discord: https://discord.com/channels/1104007013329014884.

Best of luck!

AAICO team.
