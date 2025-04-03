# dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, root, year='2007', image_set='trainval', transforms=None, selected_classes=None):
        """
        Custom Dataset for PASCAL VOC 2007.

        Args:
            root (str): Root directory of the VOC Dataset.
            year (str): Year of the VOC dataset (e.g., '2007').
            image_set (str): Image set to use ('train', 'trainval', 'val', 'test').
            transforms (callable, optional): Transformations to apply to the images.
            selected_classes (list, optional): List of classes to include.
        """
        self.voc = datasets.VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transforms = transforms
        self.selected_classes = selected_classes if selected_classes is not None else []
        self.class_map = {cls: idx for idx, cls in enumerate(self.selected_classes)}

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        """
        Retrieve the image and its corresponding targets.

        Args:
            idx (int): Index of the data sample.

        Returns:
            image (Tensor): Transformed image tensor.
            targets (dict): Dictionary containing bounding boxes and labels.
        """
        image, annotation = self.voc[idx]
        # 'image' is already a PIL Image object
        # 'annotation' is a dictionary

        # Apply transformations if any
        if self.transforms:
            image = self.transforms(image)

        # Parse annotation to extract bounding boxes and labels
        targets = self.parse_annotation(annotation)

        return image, targets

    def parse_annotation(self, annotation):
        """
        Parse the annotation dictionary to extract bounding boxes and labels.

        Args:
            annotation (dict): Annotation dictionary from VOC dataset.

        Returns:
            dict: Dictionary containing tensors of bounding boxes and labels.
        """
        boxes = []
        labels = []

        # Check if 'object' is a list or dict (single object case)
        objs = annotation['annotation'].get('object', [])
        if isinstance(objs, dict):
            objs = [objs]

        for obj in objs:
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

        # Handle images without any selected classes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return {'boxes': boxes, 'labels': labels}

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

# Create data loaders with num_workers=0 for easier debugging
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,  # Set to 0 for easier debugging
    collate_fn=lambda x: tuple(zip(*x))
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,  # Set to 0 for easier debugging
    collate_fn=lambda x: tuple(zip(*x))
)