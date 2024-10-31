import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Output: (batch_size, 32, 224, 224)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: (batch_size, 64, 112, 112)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: (batch_size, 128, 56, 56)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions by 2
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # Adjust the input size based on the output from conv layers
        self.fc2 = nn.Linear(256, num_classes)
        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: (batch_size, 32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (batch_size, 64, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # Output: (batch_size, 128, 28, 28)
        # Flatten the tensor
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(x)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    # def train(self, train_loader, valid_loader, criterion, optimizer, device='cuda'):
    #     # Training loop
    #     num_epochs = 10

    #     for epoch in range(num_epochs):
    #         self.train()
    #         running_loss = 0.0
    #         correct = 0
    #         total = 0
    #         for inputs, labels in train_loader:
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)

    #             optimizer.zero_grad()

    #             outputs = self(inputs)
    #             loss = criterion(outputs, labels)

    #             loss.backward()
    #             optimizer.step()

    #             running_loss += loss.item() * inputs.size(0)

    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #         epoch_loss = running_loss / len(train_loader.dataset)
    #         epoch_acc = 100 * correct / total
    #         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    #         # Validation
    #         self.eval()
    #         val_loss = 0.0
    #         val_correct = 0
    #         val_total = 0
    #         with torch.no_grad():
    #             for inputs, labels in valid_loader:
    #                 inputs = inputs.to(device)
    #                 labels = labels.to(device)

    #                 outputs = self(inputs)
    #                 loss = criterion(outputs, labels)

    #                 val_loss += loss.item() * inputs.size(0)

    #                 _, predicted = torch.max(outputs.data, 1)
    #                 val_total += labels.size(0)
    #                 val_correct += (predicted == labels).sum().item()

    #         val_epoch_loss = val_loss / len(valid_loader.dataset)
    #         val_epoch_acc = 100 * val_correct / val_total
    #         print(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%')

    # def test(self, test_loader, device='cuda'):
    #     # Testing
    #     self.eval()
    #     test_correct = 0
    #     test_total = 0
    #     with torch.no_grad():
    #         for inputs, labels in test_loader:
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)

    #             outputs = self(inputs)

    #             _, predicted = torch.max(outputs.data, 1)
    #             test_total += labels.size(0)
    #             test_correct += (predicted == labels).sum().item()

    #     test_acc = 100 * test_correct / test_total
    #     print(f'Test Accuracy: {test_acc:.2f}%')