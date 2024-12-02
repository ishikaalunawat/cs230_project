import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNN(nn.Module):
    """
    Denoising Convolutional Neural Network (DnCNN).
    """
    def __init__(self, depth=17, in_channels=3, out_channels=3, num_features=64):
        super(DnCNN, self).__init__()
        layers = []

        # first layer
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # mid layers
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # last layer
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.dncnn(x)
        return x - residual  # predict noise and remove it from input

