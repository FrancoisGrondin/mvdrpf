from torch.utils.data import Dataset
import librosa as lr
import numpy as np
import os

class SpexIrm(Dataset):
    
    def __init__(self, path, frame_size=512, hop_size=128, epsilon=1e-20, beta=1.0, channel_count=2):
    
        # Load list of all files
        with open(path, 'r') as file:
            self.files = file.readlines()

        # Save parameters
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.epsilon = epsilon
        self.beta = beta
        self.channel_count = channel_count

    def __len__(self):        

        return len(self.files)

    def __getitem__(self, idx):

        # Get path
        file = self.files[idx].rstrip('\n')

        # Load in time domain
        x, _ = lr.load(file, sr=16000, mono=False)
        
        # Channel 0 = target, channel 1 = interference, channel 2 = reference
        x_target = x[0, :]
        x_interf = x[1, :]
        x_ref = x[2, :]

        # Compute STFT for each signal: F x T
        X_target = lr.stft(x_target, n_fft=self.frame_size, hop_length=self.hop_size)
        X_interf = lr.stft(x_interf, n_fft=self.frame_size, hop_length=self.hop_size)
        X_ref = lr.stft(x_ref, n_fft=self.frame_size, hop_length=self.hop_size)

        # Frame index first, bin index second: T x F
        X_target = np.swapaxes(X_target, axis1=0, axis2=1)
        X_interf = np.swapaxes(X_interf, axis1=0, axis2=1)
        X_ref = np.swapaxes(X_ref, axis1=0, axis2=1)

        # Set in format: T x F x 2
        if self.channel_count == 1:
            X_cat = np.concatenate((np.expand_dims(X_target, axis=2), np.expand_dims(X_target, axis=2)), axis=2)
        if self.channel_count == 2:
            X_cat = np.concatenate((np.expand_dims(X_target, axis=2), np.expand_dims(X_interf, axis=2)), axis=2)
        X = np.log(np.abs(X_cat) + self.epsilon)

        # Compute ideal ratio mask: T x F
        M = np.clip(np.abs(X_ref) / (np.abs(X_target) + self.epsilon), a_min=0.0, a_max=1.0)

        # Return weighting time-frequency factor
        W = np.abs(X_target) ** self.beta

        # Return STFT for time-domain reconstruction
        Y = np.concatenate((np.expand_dims(X_target, axis=2),
                            np.expand_dims(X_ref, axis=2)), axis=2)

        return X, M, W, Y