import os
import sys
import argparse
import json

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, set_seed
from src.data.mnist_dataset import get_dataloaders, compute_class_distribution
from src.visualization.plots import plot_class_distribution, plot_sample_images

def main():
    parser = argparse.ArgumentParser(description="Prepare MNIST Dataset")
    parser.add_argument('--config', type=str, default='configs/mnist_config.yaml', help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(getattr(getattr(config, 'training', None), 'seed', 42))

    print("Loading data and creating splits...")
    train_loader, val_loader, test_loader, train_labels, val_labels, test_labels = get_dataloaders(config)

    print("\nComputing class distributions...")
    train_dist = compute_class_distribution(train_labels, "train")
    val_dist = compute_class_distribution(val_labels, "val")
    test_dist = compute_class_distribution(test_labels, "test")

    print("\nGenerating plots...")
    figures_dir = getattr(config.paths, 'figures_dir', './outputs/figures')
    
    plot_class_distribution(train_dist, "Training Set Class Distribution", os.path.join(figures_dir, "train_dist.png"))
    plot_class_distribution(val_dist, "Validation Set Class Distribution", os.path.join(figures_dir, "val_dist.png"))
    plot_class_distribution(test_dist, "Test Set Class Distribution", os.path.join(figures_dir, "test_dist.png"))
    
    print("Plotting sample images from training set...")
    plot_sample_images(train_loader.dataset, 5, os.path.join(figures_dir, "sample_images.png"))

    print("\nSaving split statistics...")
    output_dir = getattr(config.paths, 'output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        'train_samples': len(train_labels),
        'val_samples': len(val_labels),
        'test_samples': len(test_labels),
        'train_dist': train_dist,
        'val_dist': val_dist,
        'test_dist': test_dist
    }
    
    with open(os.path.join(output_dir, 'split_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    print("\nData pipeline preparation complete!")

if __name__ == '__main__':
    main()
