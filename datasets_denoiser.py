import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class AquariumDataset(Dataset):
    def __init__(self, images_dir, noisy_dir, transform=None):
        self.images_dir = images_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))
        self.noisy_files = sorted(os.listdir(noisy_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        clean_image_path = os.path.join(self.images_dir, self.image_files[idx])
        noisy_image_path = os.path.join(self.noisy_dir, self.noisy_files[idx])

        clean_image = Image.open(clean_image_path).convert("RGB")
        noisy_image = Image.open(noisy_image_path).convert("RGB")

        if self.transform:
            clean_image = self.transform(clean_image)
            noisy_image = self.transform(noisy_image)

        return {"clean": clean_image, "noisy": noisy_image}
