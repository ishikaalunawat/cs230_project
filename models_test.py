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


class EnhancedDnCNN(nn.Module):
    """
    Enhanced DnCNN for Denoising and Deblurring.
    """
    def __init__(self, depth=20, in_channels=3, out_channels=3, num_features=64):
        super(EnhancedDnCNN, self).__init__()
        self.noise_branch = self._build_dncnn_branch(depth, in_channels, out_channels, num_features)
        self.sharpness_branch = self._build_dncnn_branch(depth, in_channels, out_channels, num_features)
        
        # Final refinement layer
        self.refinement = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)

    def _build_dncnn_branch(self, depth, in_channels, out_channels, num_features):
        """
        Builds a DnCNN branch for either noise removal or sharpness restoration.
        """
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

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input noisy and blurred image, shape (N, C, H, W).

        Returns:
            torch.Tensor: Denoised and deblurred image, shape (N, C, H, W).
        """
        # Noise removal branch
        noise_removed = self.noise_branch(x)

        # Sharpness enhancement branch
        sharpness_enhanced = self.sharpness_branch(x)

        # Concatenate and refine
        combined = torch.cat([noise_removed, sharpness_enhanced], dim=1)
        refined = self.refinement(combined)

        return refined


# Example usage
if __name__ == "__main__":
    # Test the model with a dummy input
    model = EnhancedDnCNN(depth=20, in_channels=3, out_channels=3, num_features=64)
    x = torch.randn(1, 3, 256, 256)  # Example input image
    output = model(x)
    print("Output shape:", output.shape)

