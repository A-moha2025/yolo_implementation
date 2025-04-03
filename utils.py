# utils.py

import torch

def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, targets