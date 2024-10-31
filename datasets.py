import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Custom Dataset class for Underwater Creatures
class UnderwaterCreaturesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split  # 'train', 'valid', or 'test'
        self.transform = transform

        # Directories for images and labels
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.labels_dir = os.path.join(root_dir, split, 'labels')

        # Get list of image files
        self.image_files = os.listdir(self.images_dir)

        # Build index mapping from dataset index to (image_path, bbox, label)
        self.samples = []  # list of tuples: (image_path, bbox, label)

        for image_file in self.image_files:
            image_path = os.path.join(self.images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            bbox = [x_center, y_center, width, height]
                            self.samples.append((image_path, bbox, int(class_id)))
            else:
                # Handle images without labels if necessary
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, bbox, class_id = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # Convert bbox from normalized coordinates to pixel coordinates
        x_center, y_center, bbox_width, bbox_height = bbox
        x_center *= width
        y_center *= height
        bbox_width *= width
        bbox_height *= height

        xmin = int(x_center - bbox_width / 2)
        ymin = int(y_center - bbox_height / 2)
        xmax = int(x_center + bbox_width / 2)
        ymax = int(y_center + bbox_height / 2)

        # Crop the image to the bounding box
        cropped_image = image.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, class_id


def get_train_transform():
    # Define data transformations for training and validation/testing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225])
    ])
    return train_transform

def get_test_transform():
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
    ])
    return test_transform