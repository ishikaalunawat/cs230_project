import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import glob

def collate_fn(batch):
    images = []
    labels = []
    bboxes = []

    for item in batch:
        images.append(item[0])  # image
        labels.append(item[1])  # label
        bboxes.append(item[2])  # bounding boxes

    # stack images & labels into tensors
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    # bboxes is a list of lists (variable lengths), so we keep it as is
    return images, labels, bboxes


class UnderwaterCreaturesMultiLabelDataset(Dataset):
    def __init__(self, root_dir, split='train', num_classes=7):
        self.root_dir = root_dir
        self.split = split
        self.num_classes = num_classes

        self.images_dir = os.path.join(root_dir, split, 'images')
        self.labels_dir = os.path.join(root_dir, split, 'labels')

        self.image_files = os.listdir(self.images_dir)
        self.samples = []

        for image_file in self.image_files:
            image_path = os.path.join(self.images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_file)
            labels = torch.zeros(self.num_classes)
            bboxes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            labels[int(class_id)] = 1  # Set presence of class
                            bbox = [x_center, y_center, width, height, int(class_id)]
                            bboxes.append(bbox)
            self.samples.append((image_path, labels, bboxes))

        # Define transformations with albumentations
        if split == 'train':
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                # Add more augmentations if desired
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, labels, bboxes = self.samples[idx]
        image = np.array(Image.open(image_path).convert('RGB'))
        image_height, image_width = image.shape[:2]

        # Prepare bounding boxes and class labels
        bbox_list = []
        class_labels = []
        for bbox in bboxes:
            x_center, y_center, width, height, class_id = bbox
            bbox_list.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))

        # Apply transformations
        transformed = self.transform(image=image, bboxes=bbox_list, class_labels=class_labels)
        image = transformed['image']
        bbox_list = transformed['bboxes']
        class_labels = transformed['class_labels']

        # Combine bbox coordinates with class labels
        # Convert YOLO format back to Pascal VOC for visualization
        bbox_list_converted = []
        for bbox, class_label in zip(bbox_list, class_labels):
            x_center, y_center, width, height = bbox
            xmin = (x_center - width / 2) * 224  # Image is resized to 224x224
            ymin = (y_center - height / 2) * 224
            xmax = (x_center + width / 2) * 224
            ymax = (y_center + height / 2) * 224
            bbox_list_converted.append([xmin, ymin, xmax, ymax, class_label])

        return image, labels, bbox_list_converted

class YOLODataset(Dataset):
    def __init__(self, root_dir, split='train', num_classes=7):
        self.root_dir = root_dir
        self.split = split
        self.num_classes = num_classes

        self.images_dir = os.path.join(root_dir, split, 'images')
        self.labels_dir = os.path.join(root_dir, split, 'labels')

        self.image_files = glob.glob(os.path.join(self.images_dir, '*.jpg')) 
        self.samples = []

        for image_file in self.image_files:
            image_path = image_file
            label_file = os.path.splitext(os.path.basename(image_file))[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_file)
            labels = torch.zeros(self.num_classes)
            bboxes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            labels[int(class_id)] = 1  # Set presence of class
                            bboxes.append([x_center, y_center, width, height, int(class_id)])
            self.samples.append((image_path, labels, bboxes))

        # Define transformations for YOLO dataset
        if split == 'train':
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(p=0.1),
                A.RandomScale(scale_limit=0.1, p=0.3),
                A.RandomCrop(width=224, height=224, p=0.3),
                A.Rotate(limit=15, p=0.3),
                A.HueSaturationValue(p=0.3),
                A.ElasticTransform(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, labels, bboxes = self.samples[idx]
        image = np.array(Image.open(image_path).convert('RGB'))

        # Prepare bounding boxes and class labels
        bbox_list = []
        class_labels = []
        for bbox in bboxes:
            x_center, y_center, width, height, class_id = bbox
            bbox_list.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))

        # Apply transformations
        transformed = self.transform(image=image, bboxes=bbox_list, class_labels=class_labels)
        image = transformed['image']
        bbox_list = transformed['bboxes']
        class_labels = transformed['class_labels']

        # Convert bounding boxes to proper format for visualization
        bbox_list_converted = []
        for bbox, class_label in zip(bbox_list, class_labels):
            x_center, y_center, width, height = bbox
            xmin = (x_center - width / 2) * 224  # Image is resized to 224x224
            ymin = (y_center - height / 2) * 224
            xmax = (x_center + width / 2) * 224
            ymax = (y_center + height / 2) * 224
            bbox_list_converted.append([xmin, ymin, xmax, ymax, class_label])

        return image, labels, bbox_list_converted
