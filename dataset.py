# dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
import xml.etree.ElementTree as ET
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, root, year='2007', image_set='trainval', transforms=None, selected_classes=None):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transforms = transforms
        self.selected_classes = selected_classes

        # Initialize VOCDetection
        voc = datasets.VOCDetection(root=self.root, year=self.year, image_set=self.image_set, download=True)
        
        # Extract image and annotation file paths
        self.images = voc.images  # List of image file paths
        self.annotations = voc.annotations  # List of annotation objects
        
        # Mapping of class names to indices
        self.class_names = selected_classes
        self.class_map = {cls: idx for idx, cls in enumerate(self.class_names)}
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image using file path
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        # Load corresponding annotation
        annotation = self.annotations[idx]
        targets = self.parse_annotation(annotation)

        # Apply transformations if any
        if self.transforms:
            image = self.transforms(image)

        return image, targets
    
    def parse_annotation(self, annotation):
        boxes = []
        labels = []
        for obj in annotation['annotation']['object']:
            cls = obj['name']
            if cls in self.class_map:
                # Extract bounding box coordinates
                bbox = obj['bndbox']
                xmin = float(bbox['xmin'])
                ymin = float(bbox['ymin'])
                xmax = float(bbox['xmax'])
                ymax = float(bbox['ymax'])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_map[cls])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        return target

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Selected classes
selected_classes = ["person", "bicycle", "car", "dog", "chair"]

# Initialize dataset
dataset = VOCDataset(root='.', year='2007', image_set='trainval', transforms=transform, selected_classes=selected_classes)

# Split into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))