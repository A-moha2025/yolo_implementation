# Simplified YOLO Implementation for CIS 313 AI Project - aljouf university 

## Overview

This repository contains a **simplified implementation of the YOLO (You Only Look Once)** real-time object detection algorithm tailored for educational purposes. The model is trained on a subset of the **PASCAL VOC 2007** dataset, focusing on **5 object classes** to ensure manageability and efficiency.

## Project Structure

```
yolo_implementation/
├── dataset.py              # Dataset handling and preprocessing
├── utils.py                # Utility functions (e.g., collate_fn)
├── model.py                # Simplified YOLO model architecture
├── loss.py                 # YOLO loss function
├── train.py                # Training script
├── inference.py            # Inference and visualization script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── checkpoints/            # Saved model checkpoints
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yolo_implementation.git
cd yolo_implementation
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv yolovenv
source yolovenv/bin/activate  # On Windows: yolovenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The project utilizes the **PASCAL VOC 2007** dataset. The `dataset.py` script handles downloading and preprocessing the data for training and validation.

### Running Dataset Preparation

No additional steps are required. The dataset will be downloaded automatically when running the training script.

## Training the Model

Train the simplified YOLO model using the `train.py` script.

### Command

```bash
python train.py
```

### Parameters

- `--num_epochs`: Number of training epochs (default: 20)
- `--learning_rate`: Learning rate for the optimizer (default: 1e-3)
- `--device`: Device to run training on (`cpu` or `cuda`) (default: `cpu`)

### Example

```bash
python train.py --num_epochs 30 --learning_rate 0.001 --device cuda
```

> **Note:** Training on a GPU (`cuda`) is recommended for faster convergence.

## Model Checkpoints

Model weights are saved in the `checkpoints/` directory after each epoch (e.g., `yolov_simple_epoch_20.pth`).

## Inference and Visualization

Perform object detection on new images using the trained model with the `inference.py` script.

### Command

```bash
python inference.py --model checkpoints/yolov_simple_epoch_20.pth --image path/to/image.jpg --device cpu
```

### Parameters

- `--model`: Path to the trained model checkpoint.
- `--image`: Path to the input image for detection.
- `--device`: Device to run inference on (`cpu` or `cuda`) (default: `cpu`)

### Example

```bash
python inference.py --model checkpoints/yolov_simple_epoch_20.pth --image images/test.jpg --device cuda
```

### Output

The script displays the input image with detected bounding boxes and class labels overlaid.

## Customization

### Selecting Different Classes

To modify the object classes, update the `selected_classes` list in `dataset.py` with desired classes from **PASCAL VOC**.

### Adjusting Grid Size

Change the grid size `S` in both `model.py` and `loss.py` to experiment with different resolutions.

## Troubleshooting

- **Data Not Downloading:** Ensure an active internet connection during the first run.
- **CUDA Errors:** Verify CUDA installation and PyTorch's CUDA compatibility.
- **Insufficient Memory:** Reduce the batch size in `train.py` if encountering memory issues.
