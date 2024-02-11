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

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Initial convolution layer to extract low-level features from the input spectrogram.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # Subsequent layers increase the complexity of features extracted.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        #Depthwise separable convolution to reduce computational cost without sacrificing depth.
        self.depthwise = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pointwise = nn.Conv2d(32, 64, kernel_size=1)
        
        # Normalization and activation for depthwise and pointwise layers.
        self.bn3_depthwise = nn.BatchNorm2d(32)
        self.bn3_pointwise = nn.BatchNorm2d(64)
        
        # Global Average Pooling to reduce overfitting by minimizing the total number of parameters.
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        # Forward pass through the network.
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3_depthwise(self.depthwise(x)))
        x = self.relu(self.bn3_pointwise(self.pointwise(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer.
        x = torch.sigmoid(self.fc(x)) # Output probability for binary classification.
        return x
    
# Dataset class to manage audio frame data.
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
    

# Convert audio frame to spectrogram and normalize it.
def process(frame):
    frame_float = frame.astype(np.float32) / 32767.0
    S = librosa.feature.melspectrogram(y=frame_float, sr=sample_rate, n_fft=512, hop_length=512, n_mels=64)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_DB_tensor = torch.tensor(S_DB).unsqueeze(0).unsqueeze(0)
    # Normalize the spectrogram
    S_DB_tensor -= S_DB_tensor.mean()
    S_DB_tensor /= S_DB_tensor.std()
    return S_DB_tensor.to(device)

# Batch Processing for Frames
def process_data_batch(model, buffer, results, batch_size=10):
    start_event.wait()
    processed_batch = []
    frames_batch = []
    while not buffer.empty() or len(frames_batch) > 0:
        try:
            while len(frames_batch) < batch_size:
                frame = buffer.get_nowait()
                processed_frame = process(frame)
                frames_batch.append(processed_frame)
        except queue.Empty:
            pass

        if len(frames_batch) > 0:
            batch_tensor = torch.stack(frames_batch).to(device)
            batch_predictions = predict(model, batch_tensor)
            processed_batch.extend(batch_predictions)
            frames_batch = []


# Model instantiation and moving to the appropriate device
model = CustomCNN().to(device)


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
        outputs = model(processed_data)
        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(outputs, dim=1)
        # Get the predicted class (0 or 1)
        _, predicted_labels = torch.max(probabilities, 1)
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