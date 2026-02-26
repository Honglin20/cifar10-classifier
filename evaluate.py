"""
Evaluation Script for CIFAR-10 Classifier
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from model import get_model


# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']


def get_test_loader(batch_size=128, num_workers=4):
    """Create test data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    return DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )


def evaluate_model(model, loader, device):
    """Evaluate model and return accuracy and per-class accuracy."""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Per-class accuracy
            for i in range(len(targets)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == targets[i]).item()
                class_total[label] += 1
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    overall_acc = 100. * correct / total
    per_class_acc = [100. * c / t if t > 0 else 0 
                     for c, t in zip(class_correct, class_total)]
    
    return overall_acc, per_class_acc, all_preds, all_targets


def plot_results(per_class_acc, save_path='results.png'):
    """Plot per-class accuracy."""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(CLASSES, per_class_acc, color='steelblue', edgecolor='black')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('CIFAR-10 Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Results plot saved to {save_path}")
    plt.close()


def visualize_predictions(model, loader, device, num_images=16, save_path='predictions.png'):
    """Visualize model predictions on sample images."""
    model.eval()
    
    # Get a batch of images
    data_iter = iter(loader)
    images, labels = next(data_iter)
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Make predictions
    with torch.no_grad():
        images_device = images.to(device)
        outputs = model(images_device)
        _, predicted = outputs.max(1)
    
    # Denormalize images for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    images_display = images * std + mean
    images_display = torch.clamp(images_display, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Model Predictions (Green=Correct, Red=Wrong)', fontsize=14, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < num_images:
            img = images_display[idx].permute(1, 2, 0).numpy()
            ax.imshow(img)
            
            pred_class = CLASSES[predicted[idx]]
            true_class = CLASSES[labels[idx]]
            is_correct = predicted[idx] == labels[idx]
            
            color = 'green' if is_correct else 'red'
            ax.set_title(f'Pred: {pred_class}\nTrue: {true_class}', 
                        color=color, fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Predictions visualization saved to {save_path}")
    plt.close()


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = get_model(num_classes=10, device=device)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python train.py")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load test data
    print("Loading test data...")
    test_loader = get_test_loader()
    
    # Evaluate
    print("\nEvaluating model...")
    overall_acc, per_class_acc, preds, targets = evaluate_model(model, test_loader, device)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Overall Test Accuracy: {overall_acc:.2f}%")
    print(f"{'='*50}")
    print("\nPer-Class Accuracy:")
    for cls, acc in zip(CLASSES, per_class_acc):
        print(f"  {cls:12s}: {acc:6.2f}%")
    print(f"{'='*50}")
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_results(per_class_acc)
    visualize_predictions(model, test_loader, device)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
