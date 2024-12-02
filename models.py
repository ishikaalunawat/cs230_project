import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class DenoiserUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DenoiserUNet, self).__init__()
        # encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # SE blocks
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        # pooling and dropout
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.1)

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

        # final layer
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # encode
        enc1 = self.enc1(x)
        enc1 = self.se1(enc1)
        
        enc2 = self.enc2(self.pool(enc1))
        enc2 = self.se2(enc2)
        
        enc3 = self.enc3(self.pool(enc2))
        enc3 = self.se3(enc3)
        
        enc4 = self.enc4(self.pool(enc3))
        enc4 = self.se4(enc4)

        # bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        bottleneck = self.dropout(bottleneck)

        # decode
        dec4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))
        dec4 = self.dropout(dec4)

        dec3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))

        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        return self.final(dec1)