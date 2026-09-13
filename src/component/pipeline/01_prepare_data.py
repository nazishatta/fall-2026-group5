import os
import sys
import argparse
import json
from pathlib import Path

# Project root: C:\GIT\Capstone-2026
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.component.utils.config import load_config, set_seed
from src.component.data.mnist_dataset import (
    get_dataloaders,
    compute_class_distribution
)
from src.component.visualization.plots import (
    plot_class_distribution,
    plot_sample_images
)

def main():
    parser = argparse.ArgumentParser(description="Prepare MNIST Dataset")
    parser.add_argument(
        '--config',
        type=str,
        default=str(PROJECT_ROOT / 'src' / 'component' / 'configs' / 'week_2_baseline.yaml'),
        help='Path to config file (default: week_2_baseline.yaml)'
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
    figures_dir = getattr(config.paths, 'figures_dir', './output/week_2/figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_class_distribution(train_dist, "Training Set Class Distribution", os.path.join(figures_dir, "train_dist.png"))
    plot_class_distribution(val_dist, "Validation Set Class Distribution", os.path.join(figures_dir, "val_dist.png"))
    plot_class_distribution(test_dist, "Test Set Class Distribution", os.path.join(figures_dir, "test_dist.png"))
    
    print("Plotting sample images from training set...")
    plot_sample_images(train_loader.dataset, 5, os.path.join(figures_dir, "sample_images.png"))

    print("\nSaving split statistics...")
    output_dir = getattr(config.paths, 'output_dir', './output/week_2')
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
