import os
import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v1_mnist.component.utils.config import load_config, set_seed
from src.v1_mnist.component.data.mnist_dataset import (
    get_dataloaders,
    compute_class_distribution
)
from src.v1_mnist.component.visualization.cnn_baseline_plots import (
    plot_class_distribution,
    plot_sample_images
)

def main():
    parser = argparse.ArgumentParser(description="Prepare MNIST Dataset")
    parser.add_argument(
        '--config',
        type=str,
        default=str(PROJECT_ROOT / 'src' / 'v1_mnist' / 'component' / 'configs' / 'cnn_baseline.yaml'),
        help='Path to config file (default: cnn_baseline.yaml)'
    )
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
    figures_dir = getattr(config.paths, 'figures_dir', './outputs/v1_mnist/cnn_baseline/figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_class_distribution(train_dist, "Training Set Class Distribution", os.path.join(figures_dir, "train_dist.png"))
    plot_class_distribution(val_dist, "Validation Set Class Distribution", os.path.join(figures_dir, "val_dist.png"))
    plot_class_distribution(test_dist, "Test Set Class Distribution", os.path.join(figures_dir, "test_dist.png"))
    
    print("Plotting sample images from training set...")
    plot_sample_images(train_loader.dataset, 5, os.path.join(figures_dir, "sample_images.png"))

    print("\nSaving split statistics...")
    output_dir = getattr(config.paths, 'output_dir', './outputs/v1_mnist/cnn_baseline')
    metrics_dir = getattr(config.paths, 'metrics_dir', os.path.join(output_dir, 'metrics'))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    stats = {
        'train_samples': len(train_labels),
        'val_samples': len(val_labels),
        'test_samples': len(test_labels),
        'train_dist': train_dist,
        'val_dist': val_dist,
        'test_dist': test_dist
    }
    
    metrics_path = os.path.join(metrics_dir, 'split_stats.json')
    with open(metrics_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"Split statistics saved to {metrics_path}")
    print("\nData pipeline preparation complete!")

if __name__ == '__main__':
    main()
