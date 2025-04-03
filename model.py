# model.py

import torch
import torch.nn as nn

class SimpleYOLO(nn.Module):
    def __init__(self, S=7, B=1, C=5):  # 5 selected classes
        super(SimpleYOLO, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 224, 224]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # [B, 16, 112, 112]
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 112, 112]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # [B, 32, 56, 56]
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 56, 56]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # [B, 64, 28, 28]
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, 28, 28]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # [B, 128, 14, 14]
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [B, 256, 14, 14]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # [B, 256, 7, 7]
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),  # Input features: 256*7*7=12544
            nn.LeakyReLU(0.1),
            nn.Linear(1024, S * S * (B * 5 + C))  # Output: S*S*(B*5 + C) = 7*7*(5 +5)=490
        )
    
    def forward(self, x):
        x = self.features(x)
        # Debugging: Print the shape after convolutional layers
        # Remove or comment out in production
        print(f"After features: {x.shape}")  # Should be [batch_size, 256, 7, 7]
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x