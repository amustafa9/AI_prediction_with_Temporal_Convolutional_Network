# Script defines the TCN model to train the network on

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=True) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size, :].contiguous()


class TemporalBlock2d(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock2d, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        self.chomp1 = Chomp2d(padding[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        self.chomp2 = Chomp2d(padding[0])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, (1, 1), bias=True) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet2d(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet2d, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock2d(in_channels, out_channels, kernel_size, stride=1, dilation=(dilation_size, 1),
                                       padding=(((kernel_size[0] - 1) * dilation_size), 1), dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet2d(input_size, num_channels, kernel_size=(kernel_size, 3), dropout=dropout)
        self.groupnorm = nn.GroupNorm(num_channels=num_channels[-1], num_groups=1)
        self.conv = nn.Sequential(nn.Conv2d(in_channels=num_channels[-1],
                                            out_channels=20,
                                            kernel_size=(1, 3),
                                            padding=(0, 1)),
                                  nn.ReLU(),
                                  nn.GroupNorm(num_channels=20, num_groups=1),
                                  nn.Conv2d(in_channels=20,
                                            out_channels=10,
                                            kernel_size=(1, 3),
                                            padding=(0, 1)),
                                  nn.ReLU(),
                                  nn.GroupNorm(num_channels=10, num_groups=1),
                                  nn.Conv2d(in_channels=10,
                                            out_channels=1,
                                            kernel_size=(1, 3),
                                            padding=(0, 1)))

        self.synthesis = nn.Sequential(weight_norm(nn.Conv2d(in_channels=num_channels[-1],
                                                             out_channels=5,
                                                             kernel_size=(5, 3),
                                                             padding=(2, 1))),
                                       nn.ReLU(),
                                       nn.GroupNorm(num_channels=5,
                                                    num_groups=1),
                                       weight_norm(nn.Conv2d(in_channels=5,
                                                             out_channels=10,
                                                             kernel_size=(5, 3),
                                                             padding=(2, 1))),
                                       nn.ReLU(),
                                       nn.GroupNorm(num_channels=10,
                                                    num_groups=1),
                                       weight_norm(nn.Conv2d(in_channels=10,
                                                             out_channels=30,
                                                             kernel_size=(5, 3),
                                                             padding=(2, 1))))

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y = self.tcn(inputs)  # input should have dimension (N, C, H, W)
        y = self.groupnorm(y)
        out = self.conv(y).squeeze().transpose(1, 2)
        x_hat = self.synthesis(y)
        return out, x_hat


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=10, batch_first=True, num_layers=1)
        self.tcn_global = TemporalConvNet(num_inputs=1, num_channels=[3, 3, 5, 5, 6, 6, 2], kernel_size=5, dropout=0.2)
        self.tcn_local_1 = TemporalConvNet2d(num_inputs=1, num_channels=[3, 6, 6], kernel_size=(3, 3), dropout=0.2)
        self.tcn_local_2 = TemporalConvNet2d(num_inputs=1, num_channels=[3, 5, 6, 6], kernel_size=(3, 3), dropout=0.2)
        self.tcn_local_3 = TemporalConvNet2d(num_inputs=1, num_channels=[3, 5, 6, 6, 6], kernel_size=(5, 3),
                                             dropout=0.2)

        self.local_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=3,
                      kernel_size=(5, 3),
                      dilation=(1, 1),
                      padding=(2, 1)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=3,
                      out_channels=3,
                      kernel_size=(5, 3),
                      dilation=(1, 1),
                      padding=(2, 1)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=3,
                      out_channels=9,
                      kernel_size=(5, 3),
                      dilation=(2, 1),
                      padding=(4, 1)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=9,
                      out_channels=9,
                      kernel_size=(5, 3),
                      dilation=(2, 1),
                      padding=(4, 1)),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Conv2d(in_channels=9,
                      out_channels=15,
                      kernel_size=(5, 3),
                      dilation=(4, 1),
                      padding=(8, 1)),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=15,
                      out_channels=15,
                      kernel_size=(5, 3),
                      dilation=(4, 1),
                      padding=(8, 1)),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=15,
                      out_channels=15,
                      kernel_size=(5, 3),
                      dilation=(8, 1),
                      padding=(16, 1)),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=15,
                      out_channels=9,
                      kernel_size=(5, 3),
                      dilation=(8, 1),
                      padding=(16, 1)),
            nn.ReLU(),
            nn.GroupNorm(num_groups=1, num_channels=9)
        )

        self.aspp_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=15,
                      out_channels=3,
                      kernel_size=(5, 3),
                      dilation=(1, 1),
                      padding=(2, 1)),
            nn.GroupNorm(num_groups=1,
                         num_channels=3)
        )

        self.aspp_conv2 = nn.Sequential(nn.Conv2d(in_channels=15,
                                                  out_channels=3,
                                                  kernel_size=(5, 3),
                                                  dilation=(3, 1),
                                                  padding=(6, 1)
                                                  ),
                                        nn.GroupNorm(num_groups=1, num_channels=3)
                                        )

        self.aspp_conv3 = nn.Sequential(nn.Conv2d(in_channels=15,
                                                  out_channels=3,
                                                  kernel_size=(5, 3),
                                                  dilation=(6, 1),
                                                  padding=(12, 1)
                                                  ),
                                        nn.GroupNorm(num_groups=1, num_channels=3)
                                        )

        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                               out_channels=1,
                                               kernel_size=(1, 5),
                                               ))

        self.conv1x1_2 = nn.Conv1d(in_channels=6, out_channels=2, kernel_size=1)

        self.cnn_concat = nn.Conv1d(in_channels=2,
                                    out_channels=1,
                                    kernel_size=1
                                    )

        self.linear = nn.Linear(in_features=10, out_features=1)

    def forward(self, input):
        global_y = F.relu(self.tcn_global(input.squeeze()[:, :, [1]].transpose(1, 2)))
        local_y1 = F.relu(self.tcn_local_1(input))
        local_y2 = F.relu(self.tcn_local_2(input))
        local_y3 = F.relu(self.tcn_local_3(input))
        local_y = local_y1 + local_y2

        # local_y = torch.cat((local_y1, local_y2, local_y3), dim=1)
        local_y = self.conv1x1(local_y.transpose(1, 3)).squeeze().transpose(1, 2)
        # local_y = F.relu(self.conv1x1_2(local_y))
        y = local_y + global_y
        out, _ = self.lstm(y.transpose(1, 2))
        out = self.linear(out).transpose(1, 2)
        # local_features = self.local_conv_block(input)
        # local_y1 = F.relu(self.aspp_conv1(local_features))
        # local_y2 = F.relu(self.aspp_conv2(local_features))
        # local_y3 = F.relu(self.aspp_conv3(local_features))
        # local_y = torch.cat((local_y1, local_y2, local_y3), dim=1)
        # local_y = self.linear(local_features).squeeze(-1)
        # local_y = F.relu(self.conv1x1(local_y))
        # out = local_y + global_y
        # out = self.cnn_concat(out)
        return out
