import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class DenoisingPairedDataset(Dataset):
    ''' custom dataset for training the Denoiser model. 
        Input X: noisy image
        Y: ground truth image

        root_dir: directory of the dataset whose subfolders follow root_dir/train/x_path and y_path, respectively
        split: directory split
        x_path: folder name of the input X images nested in root_dir/split directory
        y_path: folder name of the input X images nested in root_dir/split directory
        transform: transform applied to dataset.
    '''
    def __init__(self, root_dir, split='train', x_path='noisy_images', y_path='images', transform=None):
        ''' Creates dataset of PIL images based on the following directory:
            root_dir/split/'images' for ground truth directory, 'noisy_images' for inputs directory.
        '''
        self.root_dir = root_dir
        self.split = split.lower()
        
        self.images_dir = os.path.join(root_dir, split, y_path)
        self.noisy_images_dir = os.path.join(root_dir, split, x_path) #defaults to noisy_images but can change it

        if not os.path.exists(self.images_dir) or not os.path.exists(self.noisy_images_dir):
            raise ValueError(f"Missing images or noisy_images directories in the path: {root_dir}/{split}")

        self.filenames = [f for f in os.listdir(self.images_dir) if os.path.exists(os.path.join(self.noisy_images_dir, f))]

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        noisy_path = os.path.join(self.noisy_images_dir, file_name)
        clean_path = os.path.join(self.images_dir, file_name)

        noisy_image = Image.open(noisy_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image
    
    def get_filename(self, idx):
        #returns the file name of a given index in a dataset.
        return self.filenames[idx]

