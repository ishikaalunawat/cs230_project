import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import AquariumDataset
from models import DenoiserUNet
from models_test import MotionAwareDenoiser
from losses import CombinedLoss
from trainer import Trainer

# configs
data_dir = "datasets_noisy/aquarium-data-cots/aquarium_pretrain"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
learning_rate = 5e-4
epochs = 30

# data transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# dataloaders
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
    "test": DataLoader( 
        AquariumDataset(
            os.path.join(data_dir, "test/images"),
            os.path.join(data_dir, "test/noisy_images"),
            transform=transform
        ), batch_size=5, shuffle=False  # 5 for viz
    )
}

# init model, loss, optimizer
# model = DenoiserUNet().to(device)
model = MotionAwareDenoiser(in_channels=3, out_channels=3, num_features=64, num_blocks=8)
loss_fn = CombinedLoss(perceptual_weight=0.1, ssim_weight=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-5)

# trainer
trainer = Trainer(model, dataloaders, optimizer, loss_fn, device, logdir="runs")

# training loop
for epoch in range(epochs):
    train_loss, train_psnr = trainer.train_epoch(epoch)
    val_loss, val_psnr = trainer.validate(epoch)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}, Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}")

# save model
torch.save(model.state_dict(), "outputs/denoiser_model.pth")

# visualize test outputs
trainer.visualize_test_outputs()

# close tensorboard
trainer.close()
