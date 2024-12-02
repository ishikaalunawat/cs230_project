import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNN(nn.Module):
    """
    Denoising Convolutional Neural Network (DnCNN).
    """
    def __init__(self, depth=17, in_channels=3, out_channels=3, num_features=64):
        """
        Args:
            depth (int): Number of layers in the network (default: 17).
            in_channels (int): Number of input channels (default: 3 for RGB images).
            out_channels (int): Number of output channels (default: 3 for RGB images).
            num_features (int): Number of feature maps per layer (default: 64).
        """
        super(DnCNN, self).__init__()
        layers = []

        # First layer
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input noisy image, shape (N, C, H, W).

        Returns:
            torch.Tensor: Denoised image, shape (N, C, H, W).
        """
        residual = self.dncnn(x)
        return x - residual  # Predict noise and subtract it from input

