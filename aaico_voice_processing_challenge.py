import librosa
import numpy as np
import time
import threading
import queue
import pickle
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# Checking for available device: GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########### PARAMETERS ###########
# DO NOT MODIFY
# Desired sample rate 16000 Hz
sample_rate = 16000
# Frame length
frame_length = 512

class SimpleAudioCNN(nn.Module):
    def __init__(self):
        super(SimpleAudioCNN, self).__init__()
        
        # Initial convolution layer with batch normalization and ReLU activation
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second convolution layer - Consider removing pooling to preserve spatial dimensions
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Third convolution layer - Continue without immediate pooling
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Use a single pooling layer to reduce dimensions before the fully connected layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Adjusted kernel size and stride
        
        # Adaptive average pooling to ensure fixed-size output
        self.ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # Fully connected layer for binary classification output
        self.fc = nn.Linear(64, 1)  # Assuming the flattened size is compatible
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        
        # Apply pooling here to manage dimensionality reduction carefully
        x = self.pool(x)
        
        x = self.ap(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))
        return x

# Model instantiation and moving to the appropriate device
model = SimpleAudioCNN().to(device)

class AudioDataset(Dataset):
    def __init__(self, frames, transform=None):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
        # Ensure the frame is of the correct shape for CNN input
        frame = np.expand_dims(frame, axis=0)  # Example: Adding channel dimension
        frame = torch.tensor(frame, dtype=torch.float32)
        return frame
    

def process(frame):
    frame_float = frame.astype(np.float32) / 32767.0  # Convert int16 to float
    # Adjust n_fft to match the frame length
    S = librosa.feature.melspectrogram(y=frame_float, sr=sample_rate, n_fft=512, hop_length=512, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_DB_tensor = torch.tensor(S_DB).unsqueeze(0).unsqueeze(0).to(device)
    return S_DB_tensor


########### AUDIO FILE ###########
# DO NOT MODIFY
# Path to the audio file
audio_file = "audio_aaico_challenge.wav"

# Read the audio file and resample it to the desired sample rate
audio_data, current_sample_rate = librosa.load(
    audio_file, 
    sr=sample_rate,
)
audio_data_int16 = (audio_data * 32767).astype(np.int16)
number_of_frames = len(audio_data_int16) // frame_length
audio_data_int16 = audio_data_int16[:number_of_frames * frame_length]
audio_duration = len(audio_data_int16) / sample_rate

def predict(model, processed_data):
    model.eval()
    with torch.no_grad():
        processed_data = processed_data.to(device)
        outputs = model(processed_data)  # Directly use the tensor
        predicted = torch.sigmoid(outputs)
        predicted_labels = (predicted > 0.5).long()
        return predicted_labels.cpu().numpy()

def preprocess_frame(frame, sr=sample_rate, n_fft=512, hop_length=512, n_mels=128):
    # Convert the raw audio frame to a mel-spectrogram
    S = librosa.feature.melspectrogram(y=frame, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB


########### STREAMING SIMULATION ###########
# DO NOT MODIFY
results = np.zeros(shape=(3, len(audio_data_int16)), dtype=np.int64)
# Detection mask lines are SENT TIME, LABEL, RECEIVE TIME. 
buffer = queue.Queue()
start_event = threading.Event()

def label_samples(list_samples_id, labels):
    receive_time = time.time_ns()
    results[1][list_samples_id] = labels
    results[2][list_samples_id] = receive_time

def notice_send_samples(list_samples_id):
    send_time = time.time_ns()
    results[0][list_samples_id] = send_time

def emit_data(): 
    time.sleep(.5)
    print('Start emitting')
    start_event.set()
    for i in range(0, number_of_frames):
        list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
        time.sleep(frame_length / sample_rate) # Simulate real time
        frame = audio_data_int16[list_samples_id]
        buffer.put(frame)
        notice_send_samples(list_samples_id)
    print('Stop emitting')

def process_data():
    i = 0
    start_event.wait()
    print('Start processing')
    while i != number_of_frames:
        frame = buffer.get()
        
        # Process the frame to prepare it for prediction
        processed_data = process(frame)
        
        # Predict the label for the processed data
        label_prediction = predict(model, processed_data)
        
        # Extract the label for the frame
        # Assuming predict function has been adjusted to accept processed_data directly
        # and returns a single label for the whole frame
        label = label_prediction[0].item()  # Convert to Python int for uniform labeling
        
        # Apply this label uniformly across the current frame's samples
        list_samples_id = np.arange(i * frame_length, (i + 1) * frame_length)
        labels = [label for _ in range(len(list_samples_id))]
        ###

        label_samples(list_samples_id, labels)
        i += 1
    print('Stop processing')
    # Save the list to a file
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)


if __name__ == "__main__": 
    time_measurement = []

    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)
    
    thread_process.start()
    thread_emit.start()