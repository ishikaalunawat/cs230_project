import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    '''
    Spatial Attention Module: Inspired by the CBAM (Convolutional Block Attention Module) paper [8] in report
    '''
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv1(x))
        return x * attention


class ScalingBlock(nn.Module):
    '''
    Multi-Scale Feature Extraction Block: Inspired by multi-scale IP techniques in DnCNN [7] from report.
    '''
    def __init__(self, in_channels, out_channels):
        super(ScalingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, stride=1)
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        fused = self.fusion(torch.cat([feat1, feat2, feat3], dim=1))
        return F.relu(fused)


class MAD(nn.Module):
    '''
    Motion-Aware Denoiser:
    Combine multi-scale feature extraction [7] + spatial attention [8]
    to address motion blur and speckle noise. The overall structure is influenced by
    principles from DnCNN [2] and CBAM [1].
    '''
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=8):
        super(MAD, self).__init__()

        # first feature extraction
        self.initial = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, stride=1)

        # scaling feature blocks
        self.blocks = nn.ModuleList([
            ScalingBlock(num_features, num_features) for _ in range(num_blocks)
        ])
        # spatial attention
        self.spatial_attention = Attention(num_features)

        # final reconstruction
        self.reconstruction = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        # first
        feat = F.relu(self.initial(x))

        # scaling blocks
        for block in self.blocks:
            feat = block(feat)

        # attention
        feat = self.spatial_attention(feat)

        # reconstruct
        output = self.reconstruction(feat)
        return output
