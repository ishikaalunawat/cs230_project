import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss.
    """
    def __init__(self, window_size=11, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted image, shape (N, C, H, W).
            target (torch.Tensor): Ground truth image, shape (N, C, H, W).

        Returns:
            torch.Tensor: SSIM loss value.
        """
        return 1 - self.ssim(pred, target)

    @staticmethod
    def gaussian_window(window_size, sigma, channels):
        """
        Creates a 2D Gaussian kernel for the given window size.
        """
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = gaussian / gaussian.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d.expand(channels, 1, window_size, window_size)
        return kernel

    def ssim(self, pred, target):
        """
        Computes SSIM between pred and target.
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        channels = pred.size(1)
        kernel = self.gaussian_window(self.window_size, 1.5, channels).to(pred.device)

        mu_pred = F.conv2d(pred, kernel, padding=self.window_size // 2, groups=channels)
        mu_target = F.conv2d(target, kernel, padding=self.window_size // 2, groups=channels)

        sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=self.window_size // 2, groups=channels) - mu_pred ** 2
        sigma_target_sq = F.conv2d(target * target, kernel, padding=self.window_size // 2, groups=channels) - mu_target ** 2
        sigma_pred_target = F.conv2d(pred * target, kernel, padding=self.window_size // 2, groups=channels) - mu_pred * mu_target

        ssim_map = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / (
                (mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        )
        if self.reduction == 'mean':
            return ssim_map.mean()
        elif self.reduction == 'sum':
            return ssim_map.sum()
        else:
            return ssim_map

class CombinedLoss(nn.Module):
    """
    Combined Loss: Pixel-wise MSE + Perceptual Loss + SSIM Loss.
    """
    def __init__(self, perceptual_weight=0.1, ssim_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight

        # Load a pre-trained VGG16 model for perceptual loss
        vgg = vgg16(pretrained=True).features[:16]  # Use first few layers
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()

    def compute_perceptual_loss(self, pred, target):
        """
        Computes perceptual loss using VGG features.
        """
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return F.mse_loss(pred_features, target_features)

    def forward(self, clean_pred, clean_target):
        """
        Combined loss calculation.
        """
        # Pixel-wise loss
        pixel_loss = self.mse_loss(clean_pred, clean_target)

        # Perceptual loss
        perceptual_loss = self.compute_perceptual_loss(clean_pred, clean_target)

        # SSIM loss
        ssim_loss = self.ssim_loss(clean_pred, clean_target)

        # Combined loss
        total_loss = (
            pixel_loss +
            self.perceptual_weight * perceptual_loss +
            self.ssim_weight * ssim_loss
        )
        return total_loss
