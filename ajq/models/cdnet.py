# Implementation of
# Alcantarilla, P. F., Stent, S., Ros, G., Arroyo, R., & Gherardi, R. (2018). Street-view change detection with deconvolutional networks. Autonomous Robots, 42(7), 1301–1322.

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 64, kernel_size=7, padding=3, stride=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=(2, 2), padding=(0, 0), return_indices=True)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=7, padding=3, stride=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=(2, 2), padding=(0, 0), return_indices=True)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=7, padding=3, stride=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 2), padding=(0, 0), return_indices=True)
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=7, padding=3, stride=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=(2, 2), padding=(0, 0), return_indices=True)
        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=7, padding=3, stride=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())

        self.upool4 = nn.MaxUnpool2d(kernel_size=2, stride=(2, 2), padding=(0, 0))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=7, padding=3, stride=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.upool3 = nn.MaxUnpool2d(kernel_size=2, stride=(2, 2), padding=(0, 0))
        self.conv7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=7, padding=3, stride=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.upool2 = nn.MaxUnpool2d(kernel_size=2, stride=(2, 2), padding=(0, 0))
        self.conv8 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=7, padding=3, stride=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.upool1 = nn.MaxUnpool2d(kernel_size=2, stride=(2, 2), padding=(0, 0))
        self.conv_out = nn.Conv2d(64, out_ch, kernel_size=7, padding=3, stride=1, bias=True)

    def forward(self, t1, t2):
        # Concatenation
        x = torch.cat([t1, t2], dim=1)

        # Contraction
        x, ind1 = self.pool1(self.conv1(x))
        x, ind2 = self.pool2(self.conv2(x))
        x, ind3 = self.pool3(self.conv3(x))
        x, ind4 = self.pool4(self.conv4(x))

        # Expansion
        x = self.conv5(self.upool4(x, ind4))
        x = self.conv6(self.upool3(x, ind3))
        x = self.conv7(self.upool2(x, ind2))
        x = self.conv8(self.upool1(x, ind1))

        # Out
        return self.conv_out(x)
