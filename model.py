import torch

class BGru(torch.nn.Module):

    def __init__(self, frame_size=512, hidden_size=128, num_layers=1):

        super(BGru, self).__init__()

        self.bn = torch.nn.BatchNorm2d(num_features=2)

        self.rnn = torch.nn.GRU(input_size=2*(int(frame_size/2+1)), 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                batch_first=True,
                                bidirectional=True)

        self.fc = torch.nn.Conv2d(in_channels=2*hidden_size,
                                  out_channels=int(frame_size/2+1),
                                  kernel_size=1)

    def forward(self, x):

        # Permute: N x T x F x 2 > N x 2 x T x F
        x = x.permute(0, 3, 1, 2)

        # Batch norm: N x 2 x T x F > N x 2 x T x F
        x = self.bn(x)

        # Permute: N x 2 x T x F > N x T x F x 2
        x = x.permute(0, 2, 3, 1)

        # View: N x T x F x 2 > N x T x 2F
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

        # RNN: N x T x 2F > N x T x 2H
        x, _ = self.rnn(x)

        # Permute: N x T x 2H > N x 2H x T
        x = x.permute(0, 2, 1)

        # Unsqueeze: N x 2H x T > N x 2H x T x 1
        x = torch.unsqueeze(x, 3)

        # Fully Connected: N x 2H x T x 1 > N x F x T x 1
        x = self.fc(x)

        # Permute: N x F x T x 1 > N x 1 x T x F
        x = x.permute(0, 3, 2, 1)

        # Squeeze: N x 1 x T x F > N x T x F
        x = torch.squeeze(x, dim=1)

        # Set between 0 and 1
        x = torch.sigmoid(x)

        return x

class UGru(torch.nn.Module):

    def __init__(self, frame_size=512, hidden_size=128, num_layers=1):

        super(UGru, self).__init__()

        self.bn = torch.nn.BatchNorm2d(num_features=2)

        self.rnn = torch.nn.GRU(input_size=2*(int(frame_size/2+1)), 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                batch_first=True,
                                bidirectional=False)

        self.fc = torch.nn.Conv2d(in_channels=hidden_size,
                                  out_channels=int(frame_size/2+1),
                                  kernel_size=1)

    def forward(self, x):

        # Permute: N x T x F x 2 > N x 2 x T x F
        x = x.permute(0, 3, 1, 2)

        # Batch norm: N x 2 x T x F > N x 2 x T x F
        x = self.bn(x)

        # Permute: N x 2 x T x F > N x T x F x 2
        x = x.permute(0, 2, 3, 1)

        # View: N x T x F x 2 > N x T x 2F
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

        # RNN: N x T x 2F > N x T x H
        x, _ = self.rnn(x)

        # Permute: N x T x H > N x H x T
        x = x.permute(0, 2, 1)

        # Unsqueeze: N x H x T > N x H x T x 1
        x = torch.unsqueeze(x, 3)

        # Fully Connected: N x H x T x 1 > N x F x T x 1
        x = self.fc(x)

        # Permute: N x F x T x 1 > N x 1 x T x F
        x = x.permute(0, 3, 2, 1)

        # Squeeze: N x 1 x T x F > N x T x F
        x = torch.squeeze(x, dim=1)

        # Set between 0 and 1
        x = torch.sigmoid(x)

        return x