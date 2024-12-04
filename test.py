import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from datasets_denoiser import AquariumDataset
from models import DenoiserUNet
from models_inspiration import MAD

# configs
data_dir = "datasets_noisy/aquarium-data-cots/aquarium_pretrain"
model_path = "outputs/denoiser_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 5  # subset for visualization

# data transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()  # scale to [0, 1]
])

# data
test_dataset = AquariumDataset(
    images_dir=os.path.join(data_dir, "valid/images"),
    noisy_dir=os.path.join(data_dir, "valid/noisy_images"),
    transform=transform
)

# subset of 5 samples for viz
test_subset = Subset(test_dataset, range(20))
# print(len(test_dataset))
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# load saved model
# model = DenoiserUNet().to(device)
# model = DnCNN(depth=17, in_channels=3, out_channels=3, num_features=64).to(device)
model = MAD(in_channels=3, out_channels=3, num_features=64, num_blocks=8).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()


def visualize_predictions(model, test_loader):
    for idx, batch in enumerate(test_loader):
        print("Batch: ", idx)
        noisy_images = batch["noisy"].to(device)
        clean_images = batch["clean"].to(device)

        with torch.no_grad():
            outputs = model(noisy_images)

        # convert to correct shape
        noisy_images = noisy_images.cpu().numpy().transpose(0, 2, 3, 1)
        outputs = outputs.cpu().numpy().transpose(0, 2, 3, 1)
        clean_images = clean_images.cpu().numpy().transpose(0, 2, 3, 1)

        # denormalize
        noisy_images = np.clip(noisy_images, 0, 1)
        outputs = np.clip(outputs, 0, 1)
        clean_images = np.clip(clean_images, 0, 1)

        # plot
        fig, axes = plt.subplots(5, 3, figsize=(12, 15))
        for i in range(5):
            axes[i, 0].imshow(noisy_images[i])
            axes[i, 0].set_title("Input (Noisy)")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(outputs[i])
            axes[i, 1].set_title("Model Output")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(clean_images[i])
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(f"outputs/valid_visualization_batch_{idx}.png")
        plt.show()

# run viz
visualize_predictions(model, test_loader)
