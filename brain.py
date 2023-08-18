import torch
import progressbar
import matplotlib.pyplot as plt
import os

class Brain:

    def __init__(self, net, dset):

        torch.backends.cudnn.enabled = True
        use_cuda = torch.cuda.is_available()
        
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.net = net.to(self.device)
        self.dset = dset
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
            for X, M in progressbar.progressbar(dataloader):

                # Transfer to device
                X = X.to(self.device)
                M = M.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Predict
                M_hat = self.net(X)

                # Compute loss
                loss = self.criterion(M_hat*X[:,:,:,-1], M*X[:,:,:,-1])

                # Backprop
                loss.backward()

                # Grad descent
                self.optimizer.step()

                # Add to total
                total_loss += loss.item()

            # Compute average loss
            average_loss = total_loss / len(dataloader)

        return average_loss

    def eval(self, batch_size=1, shuffle=True, num_workers=0):

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(dataset=self.dset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers)

        # Eval mode
        self.net.eval()

        # Track total loss
        total_loss = 0.0

        # Load all batches
        for X, M in progressbar.progressbar(dataloader):

            # Transfer to device
            X = X.to(self.device)
            M = M.to(self.device)

            # Predict
            M_hat = self.net(X)

            # Compute loss
            loss = self.criterion(M_hat*X[:,:,:,-1], M*X[:,:,:,-1])

            # Add to total
            total_loss += loss.item()

        # Compute average loss
        average_loss = total_loss / len(dataloader)

        return average_loss

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
        for X, M in progressbar.progressbar(dataloader):

            # Transfer to device
            X = X.to(self.device)
            M = M.to(self.device)

            # Predict
            M_hat = self.net(X)

            # To CPU
            X = X.detach().cpu()
            M = M.detach().cpu()
            M_hat = M_hat.detach().cpu()

            # To Numpy
            X = X.numpy()
            M = M.numpy()
            M_hat = M_hat.numpy()

            # Create figure
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
