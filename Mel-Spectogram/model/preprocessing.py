from torch.utils.data import Dataset
import numpy as np
import torch
import librosa


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

# for final run
def process(frame):
    sgram = librosa.stft(frame.astype(float), n_fft=512)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=16000, n_fft=512)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    dataset = AudioDataset(mel_sgram.T[np.newaxis, :, :])
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dl

