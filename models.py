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
