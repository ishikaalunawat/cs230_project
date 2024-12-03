import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        # convs
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # fc
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        # dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # convs
        x = self.pool(F.relu(self.conv1(x)))  # check-x_new: (batch_size, 32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # check-x_new: (batch_size, 64, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # check-x_new: (batch_size, 128, 28, 28)
        # flatten
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(x)
        # fc
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DenoisingCNN(nn.Module):
    ''' Simple CNN architecture for the first iteration of Denoising model training.
        simple encoder-decoder structure with dropout of
        0.2 initially and 0.4 at the last layer of encoder (and no dropout for the final layer of the decoder).
    '''
    def __init__(self):
        # super(DenoisingCNN, self).__init__()
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)#,
            #nn.Dropout(0.4)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DenoisingUNet(nn.Module):
    ''' Denoising CNN class with a U-Net architecture.
        3 contractive layers (encoder) and 3 expansive layers (decode).
        Final layer is Sigmoid to map pixel values to (0,1).
    '''
    def __init__(self):
        #super(DenoisingUNet, self).__init__()
        super().__init__()
        
        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc_conv1a = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # Additional layer
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.2)

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc_conv2a = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # Additional layer
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.2)

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc_conv3a = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # Additional layer
        self.dropout3 = nn.Dropout(0.4)
        
        # Decoder
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dropout4 = nn.Dropout(0.2)

        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dropout5 = nn.Dropout(0.2)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)
        x1 = self.enc_conv1a(x1)  # Pass through additional layer
        x1_pooled = self.pool1(x1)
        x1_pooled = self.dropout1(x1_pooled)

        x2 = self.enc_conv2(x1_pooled)
        x2 = self.enc_conv2a(x2)  # Pass through additional layer
        x2_pooled = self.pool2(x2)
        x2_pooled = self.dropout2(x2_pooled)

        x3 = self.enc_conv3(x2_pooled)
        x3 = self.enc_conv3a(x3)  # Pass through additional layer
        x3 = self.dropout3(x3)

        # Decoder with Skip Connections
        x4 = self.dec_conv1(x3)
        x4 = self.dropout4(x4)
        x4 = torch.cat((x4, x2), dim=1)  # Skip connection from encoder layer 2

        x5 = self.dec_conv2(x4)
        x5 = self.dropout5(x5)
        x5 = torch.cat((x5, x1), dim=1)  # Skip connection from encoder layer 1

        out = self.final_conv(x5)

        return out

class ImprovedDenoisingUNet(nn.Module):
    def __init__(self):
        #super(ImprovedDenoisingUNet, self).__init__()
        super().__init__()
        
        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=2, stride=2),  # Skip connection
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(32 + 32, 3, kernel_size=3, padding=1)  # Skip connection

    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)
        x1_pooled = self.pool1(x1)

        x2 = self.enc_conv2(x1_pooled)
        x2_pooled = self.pool2(x2)

        x3 = self.enc_conv3(x2_pooled)

        # Decoder with Skip Connections
        x4 = self.dec_conv1(x3)
        x4 = torch.cat((x4, x2), dim=1)  # Skip connection

        x5 = self.dec_conv2(x4)
        x5 = torch.cat((x5, x1), dim=1)  # Skip connection

        out = self.final_conv(x5)

        return torch.sigmoid(out)  # Map to [0, 1]

class ResidualUNet(nn.Module):
    def __init__(self):
        #super(ResidualUNet, self).__init__()
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)
        x1_pooled = self.pool(x1)

        x2 = self.enc_conv2(x1_pooled)
        x2_pooled = self.pool(x2)

        # Bottleneck
        bottleneck = self.bottleneck(x2_pooled)

        # Decoder
        x3 = self.dec_conv1(bottleneck) + x2  # Residual connection
        x4 = self.dec_conv2(x3) + x1  # Residual connection

        out = torch.sigmoid(self.final_conv(x4))
        return out
    
#Ishikaa's MotionAwareDenoiser
class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv1(x))
        return x * attention


class ScalingBlock(nn.Module):
    """
    Multi-scale block for capturing features at different resolutions.
    """
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


class MotionAwareDenoiser(nn.Module):
    """
    Motion-Aware Denoising and Deblurring Network with Spatial Attention.
    """
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=8):
        super(MotionAwareDenoiser, self).__init__()

        # first feature extraction
        self.initial = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, stride=1)

        # scaling feature block
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

