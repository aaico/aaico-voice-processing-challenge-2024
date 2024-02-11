import torch
import librosa
import numpy as np
import time
import threading
import queue
import pickle
from torch.utils.data import Dataset
from torch import nn
from torch.nn import init

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

########### PARAMETERS ###########
# DO NOT MODIFY
# Desired sample rate 16000 Hz
sample_rate = 16000
# Frame length
frame_length = 512


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()

        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), stride=stride, padding=(0, 0)
        )
        init.kaiming_normal_(self.shortcut.weight, a=0.1)
        self.shortcut.bias.data.zero_()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn(out)

        out = self.conv2(out)

        identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)

        return out


class AudioResNetClassifier(nn.Module):
    def __init__(self):
        super(AudioResNetClassifier, self).__init__()

        resnet_blocks = []

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        resnet_blocks += [self.conv1, self.relu1, self.bn1]

        resnet_blocks += [
            ResNetBlock(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ]
        resnet_blocks += [
            ResNetBlock(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ]
        resnet_blocks += [
            ResNetBlock(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ]

        self.resnet = nn.Sequential(*resnet_blocks)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        return x


def predict(model, test):
    model.eval()
    with torch.no_grad():
        for data in test:
            if torch.all(data == 0.0):
                prediction = 1
            else:
                inputs = data.to(device)

                batch_size = inputs.shape[0]
                inputs = inputs.view(batch_size, 1, 5, 128)
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s
                outputs = model(inputs)

                probs = torch.sigmoid(outputs)
                prediction = int(probs.round().item())

    return prediction


model = AudioResNetClassifier()
model.load_state_dict(torch.load("model.pkl"))
model = model.to(device)
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
audio_data_int16 = audio_data_int16[: number_of_frames * frame_length]
audio_duration = len(audio_data_int16) / sample_rate


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
    time.sleep(0.5)
    print("Start emitting")
    start_event.set()
    for i in range(0, number_of_frames):
        list_samples_id = np.arange(i * frame_length, (i + 1) * frame_length)
        time.sleep(frame_length / sample_rate)  # Simulate real time
        frame = audio_data_int16[list_samples_id]
        buffer.put(frame)
        notice_send_samples(list_samples_id)
    print("Stop emitting")


class AudioDataset(Dataset):
    def __init__(self, X, y=None):
        # Convert data to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        # Return the number of samples
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]  # Only return the input data


def process(frame):
    spectrogram = librosa.stft(frame.astype(float), n_fft=512)
    spectrogram_mag, _ = librosa.magphase(spectrogram)
    m_scale_spectrogram = librosa.feature.melspectrogram(
        S=spectrogram_mag, sr=sample_rate, n_fft=512
    )
    m_spectrogram = librosa.amplitude_to_db(m_scale_spectrogram, ref=np.min)
    dataset = AudioDataset(m_spectrogram.T[np.newaxis, :, :])
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dl


def process_data():
    i = 0
    start_event.wait()
    print("Start processing")
    while i != number_of_frames:
        frame = buffer.get()

        data = process(frame)
        lab = predict(model, data)
        print(lab)
        list_samples_id = np.arange(i * frame_length, (i + 1) * frame_length)
        labels = [lab for _ in range(len(list_samples_id))]

        label_samples(list_samples_id, labels)
        i += 1
    print("Stop processing")
    # Save the list to a file
    with open("results.pkl", "wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    time_measurement = []

    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)

    thread_process.start()
    thread_emit.start()
