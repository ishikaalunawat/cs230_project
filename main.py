import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import AquariumDataset
from models import DenoiserUNet
from models_test import DnCNN
from losses import CombinedLoss
from trainer import Trainer

# Configuration
data_dir = "datasets_noisy/aquarium-data-cots/aquarium_pretrain"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
learning_rate = 1e-4
epochs = 10

# Data preparation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataloaders = {
    "train": DataLoader(
        AquariumDataset(
            os.path.join(data_dir, "train/images"),
            os.path.join(data_dir, "train/noisy_images"),
            transform=transform
        ), batch_size=batch_size, shuffle=True
    ),
    "valid": DataLoader(
        AquariumDataset(
            os.path.join(data_dir, "valid/images"),
            os.path.join(data_dir, "valid/noisy_images"),
            transform=transform
        ), batch_size=batch_size, shuffle=False
    ),
    "test": DataLoader(  # Added test DataLoader for visualization
        AquariumDataset(
            os.path.join(data_dir, "test/images"),
            os.path.join(data_dir, "test/noisy_images"),
            transform=transform
        ), batch_size=5, shuffle=False  # Batch size is 5 for visualization
    )
}

# Model, loss, optimizer
model = DenoiserUNet().to(device)
# model = DnCNN(depth=17, in_channels=3, out_channels=3, num_features=64).to(device)
loss_fn = CombinedLoss(perceptual_weight=0.1, ssim_weight=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Trainer
trainer = Trainer(model, dataloaders, optimizer, loss_fn, device, logdir="runs")

# Training loop
for epoch in range(epochs):
    train_loss, train_psnr = trainer.train_epoch(epoch)
    val_loss, val_psnr = trainer.validate(epoch)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}, Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}")

# Save model
torch.save(model.state_dict(), "outputs/denoiser_model.pth")

# Visualize test outputs
trainer.visualize_test_outputs()

# Close TensorBoard writer
trainer.close()
