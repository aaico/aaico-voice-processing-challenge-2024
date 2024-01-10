# AAICO January 2024 Voice Processing Challenge

## Welcome

Welcome to the AAICO January 2024 Voice Processing Challenge! This repository contains the necessary resources and information to participate in the challenge. Please read the following guidelines to ensure a smooth and successful participation.

### Challenge Overview

The challenge involves completing the '**aaico_voice_processing_challenge.py**' file. This file simulates the streaming of the '**test_aaico_challenge.wav**' audio file. Frame by frame, the "emit_data" thread emits the data of the audio file. Each frame consists of 512 samples, with a sample rate of 16000 Hz for the audio file.

The "process_data" thread receives these frames. Your task is to complete the code in this thread to label each received sample and save your label using the provided function "label_samples". A sample should be labeled 0 if it is detected as a command, otherwise 1.

Once the code is executed, a '**results.pkl**' file will be saved, which is an array containing for each sample:

- The time at which the sample was emitted.
- The label you assigned to the sample.
- The time at which the sample was labelled.

You can evaluate your results directly on Colab in which the scoring method is fully explicit: https://colab.research.google.com/drive/1ekMF1UFfr3djseliJleUNpvzfyIJP57G?usp=sharing by uploading the results.pkl file (along with the test_aaico_challenge.wav file).

### Instructions

To submit your solution, clone the repository, create a branch with the name of your team, and then push your branch (see commands below). Indicate in the Solution description section (below) your team's name, the name and email of each member and a description of the solution.

```bash
git clone <repository_url>
pip install -r requirements.txt
git checkout -b your-branch-name
git commit -m "Your commit message"
git push origin your-branch-name
```

To have your solution considered, it must be reproducible locally by the "aaico" team.

### Solution description (to complete)

#### Team

Team name: [Team name]

Members:

- [Member Name] - [Member email]
- [Member Name] - [Member email]
- [Member Name] - [Member email]

#### Solution description

Provide clear and concise documentation in your code and update the README.md file with any additional information regarding your solution.

### Submission Deadline

Make sure to submit your solution before February 11th 2024, 11:59pm UAE time.

### Contact

If you have any questions or need clarification on the challenge, feel free to contact us at [theo.fagnoni@aaico.com].

Best of luck!

AAICO team.
