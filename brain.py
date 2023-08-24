import torch
import progressbar
import matplotlib.pyplot as plt
import os
import metrics
import kissdsp.io as io
import kissdsp.filterbank as fb
import numpy as np

import mir_eval
from pypesq import pesq
from pystoi import stoi

class Brain:

    def __init__(self, net, dset, dset_eval=None):

        torch.backends.cudnn.enabled = True
        use_cuda = torch.cuda.is_available()
        
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.net = net.to(self.device)
        self.dset = dset
        self.dset_eval = dset_eval
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def save_checkpoint(self, path):

        torch.save(self.net.state_dict(), path)

    def load_checkpoint(self, path):

        self.net.load_state_dict(torch.load(path))

    def train(self, batch_size=1, shuffle=True, num_workers=0, num_epochs=1):

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(dataset=self.dset, 
                                                 batch_size=batch_size, 
                                                 shuffle=shuffle, 
                                                 num_workers=num_workers)

        # Training mode
        self.net.train()

        # Average loss
        average_loss = 0.0

        # Train for number of epochs
        for epoch in range(0, num_epochs):

            # Track total loss
            total_loss = 0.0

            # Load all batches
            for X, M, W, Y in progressbar.progressbar(dataloader):

                # Transfer to device
                X = X.to(self.device)
                M = M.to(self.device)
                W = W.to(self.device)
                Y = Y.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Predict
                M_hat = self.net(X)

                # Compute loss
                X_target = X[:, :, :, 0]
                loss = self.criterion(M_hat*W, M*W)

                # Backprop
                loss.backward()

                # Grad descent
                self.optimizer.step()

                # Add to total
                total_loss += loss.item()

            # Compute average loss
            average_loss = total_loss / len(dataloader)

        return average_loss

    def eval(self, batch_size=1, shuffle=True, num_workers=0, sample_rate=16000):

        # Create dataloader
        if self.dset_eval == None:
            dataloader = torch.utils.data.DataLoader(dataset=self.dset,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     num_workers=num_workers)
        else:
            dataloader = torch.utils.data.DataLoader(dataset=self.dset_eval,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     num_workers=num_workers)

        # Eval mode
        self.net.eval()

        # Track total loss, pesq and stoi
        total_loss = 0.0
        total_pesq = 0.0
        total_stoi = 0.0
        total_sdr = 0.0
        total_metrics = 0

        # Load all batches
        for X, M, W, Y in progressbar.progressbar(dataloader):

            # Transfer to device
            X = X.to(self.device)
            M = M.to(self.device)
            W = W.to(self.device)
            Y = Y.to(self.device)

            # Predict
            M_hat = self.net(X)

            # Compute loss
            loss = self.criterion(M_hat*W, M*W)

            # Add to total loss
            total_loss += loss.item()

            # Compute evaluation metrics (PESQ, STOI, SDR)
            M = M.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
            M_hat = M_hat.detach().cpu().numpy()
            y_target_batch, y_ideal_batch, y_est_batch, y_ref_batch = metrics.timedomain(Y, M, M_hat)
            for batch_i in range(0,y_ref_batch.shape[0]):
                y_ref = y_ref_batch[batch_i,:]
                y_est = y_est_batch[batch_i,:]
                total_pesq += pesq(ref=y_ref, deg=y_est, fs=sample_rate)
                total_stoi += stoi(y_ref, y_est, fs_sig=sample_rate, extended=False)
                (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(y_ref, y_est)
                total_sdr += sdr[0]
                total_metrics += 1

        # Compute average metrics
        average_pesq = total_pesq / total_metrics
        average_stoi = total_stoi / total_metrics
        average_sdr = total_sdr / total_metrics

        # Compute average loss
        average_loss = total_loss / len(dataloader)

        return [average_loss, average_pesq, average_stoi, average_sdr]

    def test(self, directory):

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(dataset=self.dset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=0)

        # Eval mode
        self.net.eval()

        # Output directory
        output_directory = os.path.join(directory, '')

        # Counter to identity outputs
        counter = 0

        # Load all samples (batches of 1 sample)
        for X, M, W, Y in progressbar.progressbar(dataloader):

            # Transfer to device
            X = X.to(self.device)
            M = M.to(self.device)
            W = W.to(self.device)                    
            Y = Y.to(self.device)

            # Predict
            M_hat = self.net(X)

            # To CPU
            X = X.detach().cpu()
            M = M.detach().cpu()
            W = W.detach().cpu()
            Y = Y.detach().cpu()
            M_hat = M_hat.detach().cpu()

            # To Numpy
            X = X.numpy()
            M = M.numpy()
            W = W.numpy()
            Y = Y.numpy()
            M_hat = M_hat.numpy()

            # Back to time-domain
            y_target, y_ideal, y_est, y_ref = metrics.timedomain(Y, M, M_hat)
            ys = np.concatenate((y_target, y_ideal, y_est, y_ref), axis=0)

            # File name
            path = "%s%08u.wav" % (output_directory, counter)

            # Save
            io.write(ys, path)

            # Create figure
            plt.clf()
            plt.subplot(4, 1, 1)
            plt.imshow(X[0, :, :, 0].T, aspect='auto', origin='lower')
            plt.subplot(4, 1, 2)
            plt.imshow(X[0, :, :, 1].T, aspect='auto', origin='lower')
            plt.subplot(4, 1, 3)
            plt.imshow(M[0, :, :].T, aspect='auto', origin='lower')
            plt.subplot(4, 1, 4)
            plt.imshow(M_hat[0, :, :].T, aspect='auto', origin='lower')

            # File name
            path = "%s%08u.png" % (output_directory, counter)

            # Save
            plt.savefig(path)

            counter += 1
