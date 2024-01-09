# AAICO January 2024 Voice Processing Challenge

## Welcome

Welcome to the AAICO January 2024 Voice Processing Challenge! This repository contains the necessary resources and information to participate in the challenge. Please read the following guidelines to ensure a smooth and successful participation.

### Challenge Overview

The challenge involves completing the '**aaico_voice_processing_challenge.py**' file. This file simulates the streaming of the '**test_aaico_challenge.wav**' audio file. Frame by frame, the "emit_data" thread emits the data of the audio file. Each frame consists of 512 samples, with a sample rate of 16000 Hz for the audio file.

The "process_data" thread receives these frames. Your task is to complete the code in this thread to classify each received sample by modifying the "detection_mask" array. A sample will be labeled 1 if it is detected as communication (to be broadcasted), otherwise 0.

Once the code is executed, a '**results.pkl**' file will be saved. It will contain:

- A time tuple (time between the start of streaming and the end of processing).
- The "detection_mask" array.

You can evaluate your results directly on Colab: https://colab.research.google.com/drive/1ekMF1UFfr3djseliJleUNpvzfyIJP57G#scrollTo=YPpbjTAkDTRj by uploading the results.pkl file (along with the test_aaico_challenge.wav file).

### Instructions

```bash
git clone <repository_url>
pip install -r requirements.txt
git checkout -b your-branch-name
git commit -m "Your commit message"
git push origin your-branch-name
```

### Solution description (to complete)

Provide clear and concise documentation in your code and update the README.md file with any additional information regarding your solution.

### Submission Deadline

Make sure to submit your solution before February 11th 2024, 11:59pm UAE time.

### Contact

If you have any questions or need clarification on the challenge, feel free to contact us at [theo.fagnoni@aaico.com].

Best of luck!

AAICO team.
