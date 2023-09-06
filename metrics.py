import librosa as lr
import numpy as np
import torch

def timedomain(Y, M, M_hat, hop_size=128):

    # Get STFTs
    Y_target = Y[:, :, :, 0]
    Y_ref = Y[:, :, :, 1]

    # Apply estimated and ideal masks
    Y_ideal = Y_target * M
    Y_est = Y_target * M_hat

    # Bin index first, frame index second: N x T x F > N x T x F
    Y_target = np.swapaxes(Y_target, axis1=1, axis2=2) 
    Y_ideal = np.swapaxes(Y_ideal, axis1=1, axis2=2)
    Y_est = np.swapaxes(Y_est, axis1=1, axis2=2)
    Y_ref = np.swapaxes(Y_ref, axis1=1, axis2=2)

    # Back to time-domain
    y_target = lr.istft(Y_target, hop_length=hop_size)
    y_ideal = lr.istft(Y_ideal, hop_length=hop_size)
    y_est = lr.istft(Y_est, hop_length=hop_size)
    y_ref = lr.istft(Y_ref, hop_length=hop_size)

    return y_target, y_ideal, y_est, y_ref


def timedomain_torch(Y, M, M_hat, frame_size=512, hop_size=128):

    # Get STFTs
    Y_target = Y[:, :, :, 0]
    Y_ref = Y[:, :, :, 1]

    # Apply estimated and ideal masks
    Y_ideal = Y_target * M
    Y_est = Y_target * M_hat

    # Bin index first, frame index second: N x T x F > N x T x F
    Y_target = np.swapaxes(Y_target, axis1=1, axis2=2) 
    Y_ideal = np.swapaxes(Y_ideal, axis1=1, axis2=2)
    Y_est = np.swapaxes(Y_est, axis1=1, axis2=2)
    Y_ref = np.swapaxes(Y_ref, axis1=1, axis2=2)

    # Back to time-domain
    y_target = torch.istft(Y_target, n_fft=frame_size, hop_length=hop_size)
    y_ideal = torch.istft(Y_ideal, n_fft=frame_size, hop_length=hop_size)
    y_est = torch.istft(Y_est, n_fft=frame_size, hop_length=hop_size)
    y_ref = torch.istft(Y_ref, n_fft=frame_size, hop_length=hop_size)

    return y_target, y_ideal, y_est, y_ref