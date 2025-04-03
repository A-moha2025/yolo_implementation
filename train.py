# train.py

import torch
import torch.optim as optim
from model import SimpleYOLO
from dataset import train_loader, val_loader
from loss import YOLOLoss
from utils import collate_fn
from tqdm import tqdm

def train_model(num_epochs=20, learning_rate=1e-3, device='cpu'):
    # Initialize model, loss, optimizer
    model = SimpleYOLO().to(device)
    criterion = YOLOLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, targets in loop:
            images = images.to(device)
            # Prepare target tensor
            target_tensor = prepare_targets(targets, device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, target_tensor)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save the model checkpoint
        torch.save(model.state_dict(), f'yolov_simple_epoch_{epoch+1}.pth')

def prepare_targets(targets, device):
    """
    Convert targets into the required tensor shape.
    """
    S, B, C = 7, 1, 5
    batch_size = len(targets)
    target_tensor = torch.zeros((batch_size, S, S, B * 5 + C)).to(device)

    for i in range(batch_size):
        for box in targets[i]['boxes']:
            # Normalize coordinates
            x_center = (box[0] + box[2]) / 2.0 / 224  # Normalized to [0,1]
            y_center = (box[1] + box[3]) / 2.0 / 224
            width = (box[2] - box[0]) / 224
            height = (box[3] - box[1]) / 224

            # Determine grid cell
            grid_x = int(x_center * S)
            grid_y = int(y_center * S)

            if grid_x >= S:
                grid_x = S - 1
            if grid_y >= S:
                grid_y = S - 1

            # Assign to the first bounding box
            target_tensor[i, grid_y, grid_x, 0:2] = torch.tensor([x_center * S - grid_x, y_center * S - grid_y])
            target_tensor[i, grid_y, grid_x, 2:4] = torch.tensor([width, height])
            target_tensor[i, grid_y, grid_x, 4] = 1  # Confidence
            label = targets[i]['labels']
            if len(label) > 0:
                target_tensor[i, grid_y, grid_x, 5 + label[0]] = 1  # One-hot encoding

    return target_tensor

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(num_epochs=20, learning_rate=1e-3, device=device)