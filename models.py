import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DenoiserUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DenoiserUNet, self).__init__()
        # encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # pooling
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # final op layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # encode
        enc1 = self.enc1(x)  # [m, 64, 1024, 768]
        enc2 = self.enc2(self.pool(enc1))  # [m, 128, 512, 384]
        enc3 = self.enc3(self.pool(enc2))  # [m, 256, 256, 192]
        enc4 = self.enc4(self.pool(enc3))  # [m, 512, 128, 96]

        # middle layers
        bottleneck = self.bottleneck(self.pool(enc4))  # [m, 1024, 64, 48]

        # decode
        dec4 = self.up4(bottleneck)  # [m, 512, 128, 96]
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))  # [m, 1024, 128, 96]

        dec3 = self.up3(dec4)  # [m, 256, 256, 192]
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))  # [m, 512, 256, 192]

        dec2 = self.up2(dec3)  # [m, 128, 512, 384]
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))  # [m, 256, 512, 384]

        dec1 = self.up1(dec2)  # [n, 64, 1024, 768]
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))  # [m, 128, 1024, 768]

        # output
        out = self.out(dec1)
        return out
