# CIFAR-10 Image Classifier

A simple yet effective Convolutional Neural Network (CNN) for CIFAR-10 image classification using PyTorch.

## Features

- Clean, modular code structure
- Data augmentation for better generalization
- Training with learning rate scheduling
- Model checkpointing
- Evaluation metrics and visualization

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
numpy>=1.21.0
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

The trained model will be saved to `checkpoints/best_model.pth`.

### 3. Evaluate

```bash
python evaluate.py
```

## Model Architecture

The network consists of:
- 3 Convolutional blocks with BatchNorm and MaxPool
- 2 Fully connected layers with Dropout
- ReLU activations throughout

## Dataset

CIFAR-10 contains 60,000 32x32 color images in 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Results

Expected accuracy: ~85-88% on test set after 50 epochs.

## Project Structure

```
cifar10-classifier/
├── model.py          # CNN architecture
├── train.py          # Training script
├── evaluate.py       # Evaluation script
├── utils.py          # Helper functions
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## License

MIT
