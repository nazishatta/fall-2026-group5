import os
import sys
import argparse
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, set_seed, get_device
from src.data.mnist_dataset import get_dataloaders
from src.models.lenet5 import LeNet5
from src.features.extractor import FeatureExtractor, save_embeddings

def main():
    parser = argparse.ArgumentParser(description="Extract Features from LeNet-5 Penultimate Layer")
    parser.add_argument('--config', type=str, default='configs/mnist_config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to saved model weights')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.training.seed)
    device = get_device(config)

    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(config)

    model = LeNet5(num_classes=config.dataset.num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    layer_name = config.model.feature_layer
    print(f"Extracting features from layer: {layer_name}")
    
    extractor = FeatureExtractor(model, layer_name)
    
    embeddings_dir = os.path.join(config.paths.embeddings_dir, 'mnist')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    splits = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    for split_name, loader in splits.items():
        print(f"Extracting features for {split_name} split...")
        features, labels, preds = extractor.extract(loader, device)
        save_embeddings(features, labels, preds, split_name, embeddings_dir)
        print(f"Saved {split_name} features: shape {features.shape}, labels: shape {labels.shape}")

    print(f"\nAll embeddings saved to {embeddings_dir}")

if __name__ == '__main__':
    main()
