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

Team name: MLX

Members:

- Suhail Ahmed - suhailz13ahmed@outlook.com
- Noureldine Adib - noureldine2003@gmail.com
- Dhruv Chaturvedi - dhruvradhakant@gmail.com

#### Solution description

While popular audio machine learning probelms include speaker diarization, audio segmentation, voice activity detection (VAD) and so on and so forth, owing to the nature of the task, we approached it as a "Wake Word" problem. Popular wake word models around the world include Apple's "Siri" and Amazon's "Alexa" wherein the model does not start listening to your commands unless you use the model's corresponding wake words such as "Hey Siri" or "Alexa". Upon doing so, the model starts listening an parsing your commands.

Similarly, the probelm statement describes three commands; all starting with the word "Galactic". Since we have a small set of possible commands in this challenge (only three - oxygen, temperature and battery), we decided to leverage pre-trained models for all three commands. For the pre-trained model, we used Picovoice's (https://picovoice.ai) robust wake word model - Porcupine (https://github.com/Picovoice/porcupine). We leveraged it to train 3 different models on the words "Galactic Temperature", "Galactic Oxygen" and "Galactic Battery". The 3 pre-trianed models were later downloaded (can be found in the root directory under the folder; models), imported and used in our code.

The primary function, process_data(), processes a stream of audio frames, detects keywords using Porcupine, and labels the samples accordingly (0 or 1). We also wrote a helper function, process_label() which handles the dynamic labeling of samples based on keyword detection.

Our model not only labels commands correctly, but it is also able to distinguish between different commands. It is able to parse and identify "Galactic Oxygen", "Galactic Temperature" and "Galactic Battery" as different commands rather than treating them as the same, makign it suitable for real-world scenarios.

An overview of our two core functions can be found below:

#### process_label(start_time, end_time, flag, cur_frame, keyword=False) Function:

Parameters:

- start_time: Start time of the labeled interval.
- end_time: End time of the labeled interval.
- flag: Integer indicating the type of labeling (0 for keyword, 1 for no keyword).
- cur_frame: Current frame index.
- keyword: Boolean flag indicating if a keyword was detected in the previous frame.

Returns:

- Updated keyword flag.

- Logic:

- If flag is 0, labels samples with a detected keyword.
- If flag is 1 and the last frame was a command (keyword is True), it adjusts the labeling by rolling back onto the last parsed frame and marking it as 1 from there until the current point.
- If the flag is 1 and the last parsed frame was not a keyword, it marks it as 1.

#### process_data() Function:

Overview:

- Processes a stream of audio frames.
- Detects keywords using Porcupine.
- Calls process_label() to dynamically label samples.

Key Steps:

- Waits for the start event (start_event) before starting processing.
- Enters a loop to process each frame until the end of the stream.
- Retrieves an audio frame from the buffer (buffer.get()).
- Calls Porcupine to detect keywords (porcupine.process(frame)).
- Determines the end time of the current frame.
- Calls process_label() based on the detected keyword index.

Keyword Detection and Labeling:

- Adjusts start and end offset values based on the keyword detected.
- Utilizes process_label() to label samples dynamically.

Result:

- Saves the labeled samples and timestamps to a pickle file (results.pkl).

#### Setup

This code was developed on Python 3.11.6

- Install additional dependencies

```bash
   pip install pvporcupine
```

- Run Script

```bash
   python aaico_voice_processing_challenge.py
```

#### Results

| Constraint  | Score | 
| -------------- | --------------- | 
| Original Evaluation (With threshold penalty)      | 94.34               |
| Without slow threshold penalty    | 100              |

Due to the robust and dynamic way we appriached this problem wherein we aimed to minimize resources and maximize accuracy, we obtained an impresseive score of 94 (with the slow threshold penalty) as shown below.

<img width="620" alt="image" src="https://github.com/Suhail270/aaico-voice-processing-challenge-2024/assets/57321434/669a1dc5-ec5a-4be5-a6dc-d899e4751fcb">


If we were to comment out the threshold penalty, we get a score of 100 as shown below.

<img width="620" alt="image" src="https://github.com/Suhail270/aaico-voice-processing-challenge-2024/assets/57321434/a6b2e510-f212-47af-a607-d43210fc6084">


This highlights our succesful attemt at identifying the command, declaring offsets and labeling the emitted audio, framy by frame. The only reason we get hit with the penalty is due to us leveraging an external pre-trained model. With enough time and resources, we believe that we can develop our own in-house model to optimize our solution even further.

#### Alternate Approaches

We tried several approaches and picked the best one in terms of score (which is implemented in our current code base). Some of our discarded approaches included:

- Native CNN based on audio images - We curated a dataset of audio segments comprising the commands which we converted to images (represented as a matrix). We then compared each image segment of audio to that of our training dataset's and predict accordingly. This gave us an accuracy of 95%.

- Native Random Forest Classifier - With the same dataset along with augemntations of it (original audio samples + noisy augmentations of them), we acheived an accuracy of 93%.

However, both performed poorly on the scoring, yielding scores of only 45 and 19 respectively.

Another approach we had but did not implement is described below:

We use a wake word/keyword spotting model to identify the keyword "Galactic". Then we use a Voice Activity Detection model/pipeline to identify when the command has been said completely (non-human speech such as silence or white noise would indicate the end of the command). We would then mark the range of that as 0s (comamnds). In real-world scenarios, it may be important to distinguish between different commands as "Galactic Battery" and "Galactic Oxygen" may trigger different workflows. So it would then be important to parse the command using a speech to text/speech recognition model. However, this is a very heavy approach as it would rely on 3 different processes and models which would not be ideal for real-word scenarios. Hence, we did not attempt this.

#### Additional Notes

- The API key listed in the file is for the sake of this hackathon only and there are no issues of privacy whatsoever. We have allowance of upto 3 different users using this API key, only 1 of us is using it in our local machine. Our code runs offline as long as the API key is mentioned, we highlighted in our request that the model we trained will be used to take part in a hackathon and there may be monetary benefits out of this. We were given approval.

- On the receiver side, we label multiple frames at the same time (after we receive them) but we label each frame that was sent (using the dedicated function that feed the results array). However, due to this, the first constraint fails. We asked and verify the same in the discord chat, please find the below attached chat snippet.

![image](https://github.com/Suhail270/aaico-voice-processing-challenge-2024/assets/57321434/146715c3-4123-40b3-a5d8-79aad8874f12)

- Because we use an external pre-trained model, the time takes a longer time. Again, it was confirmed in the Discord channel that this would be taken into account and there would be some leniency.

Thank you for your understanding. 

For any queries you may have, please do contact suhailz13ahmed@outlook.com or +971547475288.

We are all students from Heriot-Watt University, Dubai.

### Submission Deadline

Make sure to submit your solution before February 11th 2024, 11:59pm UAE time.

### Contact

If you have any questions or need clarification on the challenge, feel free to reach out on Discord: https://discord.com/channels/1104007013329014884.

Best of luck!

AAICO team.
