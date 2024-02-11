from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import librosa

# Desired sample rate 16000 Hz
sample_rate = 16000

# Frame length
frame_length = 512

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


command_samples = [
    [142000, 160000],
    [340000, 360000],
    [620000, 635000]
]

nb_command_samples = sum([elem[1] - elem[0] for elem in command_samples])
ground_truth = np.ones(len(audio_data_int16))
for i in range(len(audio_data_int16)):
    if any([i >= e[0] and i <= e[1] for e in command_samples]):
        ground_truth[i] = 0

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

# for training and testing
def get_train_test():
    mel_sgrams = []

    for i in range(0, number_of_frames):
        list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
        sgram = librosa.stft(audio_data_int16[list_samples_id].astype(float), n_fft=512)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate, n_fft=512)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        mel_sgrams.append(mel_sgram.T)

    mel_sgrams = np.asarray(mel_sgrams)

    reshaped_groundtruth = ground_truth.reshape(-1, 512)
    aggregated_labels = np.mean(reshaped_groundtruth, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(mel_sgrams, aggregated_labels, test_size=0.3, shuffle=False)

    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    return train_dl, val_dl