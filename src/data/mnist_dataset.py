import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from collections import Counter

class CustomMNISTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Convert to float tensor and add channel dimension
        img = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

def compute_class_distribution(labels, split_name):
    counts = Counter(labels)
    total = len(labels)
    dist = {}
    print(f"\nClass Distribution for {split_name} split ({total} samples):")
    for cls in range(10):
        count = counts.get(cls, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        dist[cls] = {'count': count, 'percentage': percentage}
        print(f"Class {cls}: {count} samples ({percentage:.2f}%)")
    return dist

def get_dataloaders(config):
    data_root = getattr(config.paths, 'data_root', '../../All_Data')
    
    # Download/load original MNIST
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True)
    
    # Extract data and labels, convert to numpy
    train_data = train_dataset.data.numpy()
    train_labels = train_dataset.targets.numpy()
    test_data = test_dataset.data.numpy()
    test_labels = test_dataset.targets.numpy()
    
    # Combine all 70k samples
    all_data = np.concatenate((train_data, test_data), axis=0)
    all_labels = np.concatenate((train_labels, test_labels), axis=0)
    
    # Normalize pixel values to [0, 1]
    all_data = all_data / 255.0
    
    # Stratified split: 70% train, 30% temp
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        all_data, all_labels, 
        train_size=getattr(config.dataset, 'train_ratio', 0.70),
        stratify=all_labels, 
        random_state=getattr(getattr(config, 'training', None), 'seed', 42)
    )
    
    # Split temp into 50% val, 50% test (15% overall each)
    val_test_ratio = getattr(config.dataset, 'val_ratio', 0.15) / (getattr(config.dataset, 'val_ratio', 0.15) + getattr(config.dataset, 'test_ratio', 0.15))
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels,
        train_size=val_test_ratio,
        stratify=temp_labels,
        random_state=getattr(getattr(config, 'training', None), 'seed', 42)
    )
    
    train_ds = CustomMNISTDataset(train_data, train_labels)
    val_ds = CustomMNISTDataset(val_data, val_labels)
    test_ds = CustomMNISTDataset(test_data, test_labels)
    
    batch_size = getattr(config.dataloader, 'batch_size', 64)
    num_workers = getattr(config.dataloader, 'num_workers', 2)
    pin_memory = getattr(config.dataloader, 'pin_memory', True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader, train_labels, val_labels, test_labels
