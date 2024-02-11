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

Team name: V-Stream Analysts

Members:

- Mohammed Sadiq Bagalkot - sadiqshabbir4@gmail.com
- Bhavika Kaliya - bhavikakaliya@gmail.com
- Alora Tabuco - alorartabuco@gmail.com

#### Solution description

##### Submission Details

We have decided to submit two solutions: a Mel-Spec approach and a Speech-to-Text approach. The main requirements of the Hackathon were real-time processing and accurate labelling. 

Considering this, these are our two solutions.

##### Best Overall Performance: Mel-Spectrograms with CNN

Score from results.pkl (overall): **0.86**

Make sure to open code with target directory as Mel-Spectogram
To install required packages:
```
pip install -r Mel-Spectogram/requirements.txt
```
Run ```aaico_voice_processing_challenge.py``` in the Mel-Spectogram directory to get results.pkl \
Alternatively run ```run_training.py``` to train the model.

Our solution leverages Mel-Spectrograms [1] [2]  with Convolutional Neural Networks (CNN) for real-time processing and accurate labeling of audio data. Mel-Spectrograms, known for their effectiveness in representing audio as images, are utilized as input to the CNN model.

**Training**


We initiated the process by converting the provided audio files into Mel-Spectrograms. To ensure robustness, we augmented the dataset, particularly focusing on increasing samples for the 'command' (Class 0) category. Data was then split into training and testing sets, maintaining a 70/30 ratio. Notably, we refrained from shuffling the data to retain its sequential nature.

The model architecture comprises four convolutional blocks, incorporating an average pooling layer and dropout layer for regularization. We employed the BCEWithLogits loss function [3], Adam optimizer [4], and One Cycle Learning Rate scheduler [5] for training stability. The model underwent training for 30 epochs to achieve optimal performance.


**Solution in .py file**


Our solution is encapsulated within the aaico_voice_processing_challenge.py file in the Mel-Spec folder. In the real-time scenario, each emitted frame is processed by converting it into a Mel-Spectrogram. This spectrogram serves as input to the pre-trained CNN model. Subsequently, the model predicts the class label, either 'command' (Class 0) or 'communication' (Class 1), facilitating efficient processing and labeling of incoming audio data streams.

This approach combines the advantages of Mel-Spectrograms and CNNs to meet the requirements of real-time processing and accurate labeling, thereby emerging as the best-performing solution for the challenge.

##### Alternative Solution: Speech-to-Text


In addition to our primary Mel-Spectrogram solution, we present an alternative approach utilizing Facebook's pretrained model for Automatic Speech Recognition, specifically the wav2vec2-base-960h model [6]. While this model achieves accurate labeling, it falls short of meeting the real-time processing requirement (<50ms per sample).

**Implementation**


Our alternative solution is encapsulated within the `aaico_voice_processing_challenge.py` file in the Speech-to-text folder. In the real-time scenario, frames are processed in batches of 50. Every batch of 50 frames undergoes transcription. If the transcription contains the word "GALACTIC," the frames in the batch are retroactively labeled as Class 0; otherwise, they are labeled as Class 1.

**Performance**


The solution achieves an accuracy score of 0.81 when evaluated solely on accuracy metrics. However, when considering the time restriction (<50ms per sample), the performance decreases significantly to a score of 0.16. As such, while this approach provides accurate labeling, it does not fulfill the real-time processing requirement and is thus presented as an alternate approach rather than our main solution for the Hackathon problem.

### Submission Deadline

Make sure to submit your solution before February 11th 2024, 11:59pm UAE time.

### Contact

If you have any questions or need clarification on the challenge, feel free to reach out on Discord: https://discord.com/channels/1104007013329014884.

Best of luck!

AAICO team.

### References

[1]: https://en.wikipedia.org/wiki/Spectrogram 'Spectrograms'
[2]: https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html 'Mel-Spectrograms'
[3]: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html 'BCE With Logits Loss'
[4]: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html 'Adam Optimizer'
[5]: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html 'One Cycle LR'
[6]: https://huggingface.co/facebook/wav2vec2-base-960h 'Wav2Vec2 Hugging Face Transformer'
