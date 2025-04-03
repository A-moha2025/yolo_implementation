import torch
from model import SimpleYOLO
from dataset import transform
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def load_model(model_path, device='cpu'):
    model = SimpleYOLO().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(image_path, model, device='cpu'):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    transformed_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(transformed_image)

    return output, image, original_size

def visualize_predictions(output, image, original_size, threshold=0.5):
    S, B, C = 7, 1, 5
    grid_size = S
    stride_x = original_size[0] / grid_size
    stride_y = original_size[1] / grid_size

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for i in range(S):
        for j in range(S):
            cell_pred = output[0, i, j]
            confidence = cell_pred[4].item()
            if confidence > threshold:
                x_center, y_center, width, height = cell_pred[0:4].tolist()
                x_center = (x_center + j) * stride_x / S
                y_center = (y_center + i) * stride_y / S
                width = width * original_size[0] / S
                height = height * original_size[1] / S

                # Top-left corner
                xmin = x_center - width / 2
                ymin = y_center - height / 2

                # Determine class
                class_scores = cell_pred[5:]
                class_idx = torch.argmax(class_scores).item()
                class_confidence = class_scores[class_idx].item()
                label = f"Class {class_idx}: {confidence * class_confidence:.2f}"

                # Draw bounding box
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin, label, fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='YOLO Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on')

    args = parser.parse_args()

    model = load_model(args.model, device=args.device)
    output, image, original_size = predict(args.image, model, device=args.device)
    visualize_predictions(output, image, original_size, threshold=0.5)