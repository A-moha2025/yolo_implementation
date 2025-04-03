# loss.py

import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=1, C=5, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        # predictions: [Batch, S, S, B*5 + C]
        # target: [Batch, S, S, B*5 + C]

        # Coordinate loss
        coord_loss = self.mse(predictions[..., 0:2], target[..., 0:2])

        # Confidence loss
        conf_loss = self.mse(predictions[..., 4], target[..., 4])

        # Classification loss
        class_loss = self.mse(predictions[..., 5:], target[..., 5:])

        total_loss = self.lambda_coord * coord_loss + conf_loss + class_loss
        return total_loss