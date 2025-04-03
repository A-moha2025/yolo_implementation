# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleYOLO(nn.Module):
    def __init__(self, S=7, B=1, C=5):  # 5 selected classes
        super(SimpleYOLO, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 7, 7]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # [B, 16, 3, 3]
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 3, 3]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # [B, 32, 1, 1]
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 1 * 1, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, S * S * (B * 5 + C))  # 7x7 grid, 1 bounding box, 5 + 5 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x