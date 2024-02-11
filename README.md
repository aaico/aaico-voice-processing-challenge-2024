# AAICO February 2024 Voice Processing Challenge

## Solution description

### Team

Team name: OwlAI

Members:

- Nripesh Niketan - nripesh14@gmail.com
- Arunima Santhosh Kumar - arunimasanthosh2303@gmail.com

Both of us are Undergraduate students at Heriot-Watt University, Dubai Campus, pursuing a Bachelor's degree in Computer Science.

### Solution description

For the Applied AI Company (AAICO) February 2024 Hackathon, our team developed a Python-based solution to address the challenge of distinguishing between control commands and broadcast communications in a one-minute audio stream from a firefighting suit. Our approach leverages a custom AudioResNetClassifier model, optimized for speed and accuracy in real-time audio processing and wake word recognition.

#### Solution Overview

Our solution employs a deep learning model, AudioResNetClassifier, inspired by the ResNet architecture, tailored for audio signal processing. The model processes audio data frame-by-frame, with each frame converted into a Mel-spectrogram representation before being fed into the network. This approach ensures the model captures both spectral and temporal characteristics of the audio, crucial for distinguishing between voice commands and broadcast communications.

#### Model Architecture

The AudioResNetClassifier consists of an initial convolutional layer followed by a series of ResNet blocks, each designed to extract and refine features from the audio signal. The model concludes with an adaptive average pooling layer and a fully connected layer, outputting a binary classification for each audio frame: 0 for control commands and 1 for broadcast communications.

##### Key Components:

- **Initial Convolutional Layer:** Prepares the input Mel-spectrogram for deeper processing, using a 5x5 kernel.
- **ResNet Blocks:** Each block contains two convolutional layers with batch normalization and ReLU activations, including a shortcut connection to facilitate deeper learning without vanishing gradients.
- **Adaptive Average Pooling:** Reduces the feature map to a fixed size, ensuring the model remains adaptable to inputs of varying dimensions.
- **Fully Connected Layer:** Translates the high-level features extracted by the ResNet blocks into the final binary classification.

#### Training Environment

The model was trained using NVIDIA A100 GPUs on an Ubuntu machine, chosen for their computational efficiency and ability to handle large datasets rapidly. This environment enabled us to iterate quickly, experimenting with different hyperparameters and model architectures to optimize performance.

#### Dataset and Preprocessing

Training data comprised one-minute audio streams, sampled at 16 kHz, representing realistic scenarios captured from firefighting suits. Each audio stream was segmented into frames of 512 samples, converted into Mel-spectrograms, and then normalized before being inputted into the model. This preprocessing step was crucial for enhancing model sensitivity to key features indicative of voice commands versus broadcast communications.

#### Performance

Our solution achieved a score of 0.72, reflecting a balance between speed and accuracy. The model's performance was assessed based on its ability to accurately classify each audio frame with minimal processing delay, measured as the difference between the audio file's duration and the time taken for the model to process and classify the audio stream.

#### Conclusion and Future Work

This solution demonstrates the potential of deep learning in real-time audio processing applications, particularly in high-stakes environments such as firefighting. Looking ahead, we aim to further refine our model by exploring more complex architectures like AutoRegressive and training it on a broader dataset to enhance its robustness and accuracy.

By leveraging state-of-the-art hardware and a novel approach to audio stream analysis, we believe our solution represents a significant step forward in the development of intelligent voice-controlled systems, with potential applications extending beyond the scope of this hackathon.

