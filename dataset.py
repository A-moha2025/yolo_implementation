# train.py

import torch
import torch.optim as optim
from model import SimpleYOLO
from dataset import train_loader, val_loader
from loss import YOLOLoss
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model(num_epochs=20, learning_rate=1e-3, device='cpu'):
    """
    Train the SimpleYOLO model.

    Args:
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to train on ('cpu' or 'cuda').

    Returns:
        None
    """
    # Initialize model, loss, optimizer
    model = SimpleYOLO().to(device)
    criterion = YOLOLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

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
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save the model checkpoint
        torch.save(model.state_dict(), f'checkpoints/yolo_simple_epoch_{epoch+1}.pth')

    # Plotting the loss curve
    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_epochs+1), loss_history, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()

def prepare_targets(targets, device):
    """
    Convert targets into the required tensor shape.

    Args:
        targets (list of dict): List of target dictionaries for each image in the batch.
        device (torch.device): Device to place the tensor on.

    Returns:
        torch.Tensor: Tensor of shape [Batch, S, S, B*5 + C].
    """
    S, B, C = 7, 1, 5  # Grid size, bounding boxes, classes
    batch_size = len(targets)
    target_tensor = torch.zeros((batch_size, S, S, B * 5 + C)).to(device)

    for i in range(batch_size):
        for box in targets[i]['boxes']:
            # Normalize coordinates
            x_center = (box[0] + box[2]) / 2.0 / 224  # Normalize to [0,1]
            y_center = (box[1] + box[3]) / 2.0 / 224
            width = (box[2] - box[0]) / 224
            height = (box[3] - box[1]) / 224

            # Determine grid cell
            grid_x = int(x_center * S)
            grid_y = int(y_center * S)

            # Boundary check
            grid_x = min(max(grid_x, 0), S-1)
            grid_y = min(max(grid_y, 0), S-1)

            # Assign to the first bounding box (since B=1)
            target_tensor[i, grid_y, grid_x, 0:2] = torch.tensor([x_center * S - grid_x, y_center * S - grid_y])
            target_tensor[i, grid_y, grid_x, 2:4] = torch.tensor([width, height])
            target_tensor[i, grid_y, grid_x, 4] = 1  # Confidence

            # Assuming one bounding box per cell; for multiple objects, this needs to be adjusted
            labels = targets[i]['labels']
            if len(labels) > 0:
                # Assign the first label; for multiple labels in the same cell, additional logic is needed
                target_tensor[i, grid_y, grid_x, 5 + labels[0]] = 1  # One-hot encoding

    return target_tensor

if __name__ == "__main__":
    # Detect if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure the checkpoints directory exists
    import os
    os.makedirs('checkpoints', exist_ok=True)

    # Train the model
    train_model(num_epochs=20, learning_rate=1e-3, device=device)