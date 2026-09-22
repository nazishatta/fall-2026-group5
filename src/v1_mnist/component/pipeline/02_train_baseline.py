import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v1_mnist.component.utils.config import load_config, set_seed, get_device
from src.v1_mnist.component.data.mnist_dataset import get_dataloaders
from src.v1_mnist.component.models.lenet5 import LeNet5
from src.v1_mnist.component.evaluation.metrics import (
    evaluate_model,
    print_evaluation_summary
)
from src.v1_mnist.component.visualization.cnn_baseline_plots import (
    plot_training_curves,
    plot_confusion_matrix
)

def main():
    parser = argparse.ArgumentParser(description="Train LeNet-5 Baseline on MNIST")
    parser.add_argument(
        '--config',
        type=str,
        default=str(PROJECT_ROOT / 'src' / 'v1_mnist' / 'component' / 'configs' / 'cnn_baseline.yaml'),
        help='Path to config file (default: cnn_baseline.yaml)'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.training.seed)
    device = get_device(config)
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(config)

    model = LeNet5(num_classes=config.dataset.num_classes).to(device)
    print(f"Initialized LeNet-5 baseline CNN (penultimate layer: {model.feature_layer_name}, dim: {model.feature_dim})")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.step_size, gamma=config.training.gamma)

    epochs = config.training.epochs
    best_val_acc = 0.0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    checkpoints_dir = os.path.join(config.paths.output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoints_dir, 'lenet5_best.pth')

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print("Saved new best model.")

    print("\nTraining completed. Evaluating best model...")
    model.load_state_dict(torch.load(best_model_path))
    
    test_results = evaluate_model(model, test_loader, device)
    print_evaluation_summary(test_results, "test")
    
    figures_dir = getattr(config.paths, 'figures_dir', './outputs/v1_mnist/cnn_baseline/figures')
    os.makedirs(figures_dir, exist_ok=True)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, os.path.join(figures_dir, 'training_curves.png'))
    plot_confusion_matrix(test_results['confusion_matrix'], [str(i) for i in range(10)], "Test Set Confusion Matrix", os.path.join(figures_dir, 'test_confusion_matrix.png'))
    
    # Save test results
    eval_output = {
        'accuracy': test_results['accuracy'],
        'classification_report': test_results['classification_report_dict']
    }
    metrics_dir = getattr(config.paths, 'metrics_dir', os.path.join(config.paths.output_dir, 'metrics'))
    os.makedirs(metrics_dir, exist_ok=True)
    
    test_eval_path = os.path.join(metrics_dir, 'test_evaluation.json')
    with open(test_eval_path, 'w') as f:
        json.dump(eval_output, f, indent=4)
        
    print(f"Results saved to {test_eval_path}")

if __name__ == '__main__':
    main()
