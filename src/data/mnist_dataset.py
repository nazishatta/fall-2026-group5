from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


class CustomMNISTDataset(Dataset):
    """MNIST arrays with optional stable sample identifiers."""

    def __init__(self, data, labels, sample_ids=None, return_sample_id=False):
        self.data = data
        self.labels = labels
        self.sample_ids = (
            np.arange(len(data), dtype=np.int64)
            if sample_ids is None
            else np.asarray(sample_ids, dtype=np.int64)
        )
        self.return_sample_id = return_sample_id

        if not (len(self.data) == len(self.labels) == len(self.sample_ids)):
            raise ValueError("data, labels, and sample_ids must have equal lengths")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.as_tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        if self.return_sample_id:
            sample_id = torch.as_tensor(self.sample_ids[idx], dtype=torch.long)
            return image, label, sample_id
        return image, label


def compute_class_distribution(labels, split_name):
    counts = Counter(np.asarray(labels).tolist())
    total = len(labels)
    distribution = {}
    print(f"\nClass Distribution for {split_name} split ({total} samples):")
    for class_id in range(10):
        count = counts.get(class_id, 0)
        percentage = (count / total) * 100 if total else 0.0
        distribution[class_id] = {"count": count, "percentage": percentage}
        print(f"Class {class_id}: {count} samples ({percentage:.2f}%)")
    return distribution


def get_dataloaders(config, train_shuffle=True, return_sample_ids=False):
    """Create the predefined stratified 70/15/15 split.

    Defaults preserve baseline-training behavior. Official embedding extraction
    must call this with train_shuffle=False and return_sample_ids=True.

    Stable sample IDs refer to positions in the combined array: original MNIST
    training samples are 0..59999 and original MNIST test samples 60000..69999.
    """

    data_root = getattr(config.paths, "data_root", "../../All_Data")
    seed = getattr(getattr(config, "training", None), "seed", 42)

    original_train = datasets.MNIST(root=data_root, train=True, download=True)
    original_test = datasets.MNIST(root=data_root, train=False, download=True)

    all_data = np.concatenate(
        (original_train.data.numpy(), original_test.data.numpy()), axis=0
    ).astype(np.float32)
    all_data /= 255.0
    all_labels = np.concatenate(
        (original_train.targets.numpy(), original_test.targets.numpy()), axis=0
    )
    all_sample_ids = np.arange(len(all_labels), dtype=np.int64)

    train_data, temp_data, train_labels, temp_labels, train_ids, temp_ids = (
        train_test_split(
            all_data,
            all_labels,
            all_sample_ids,
            train_size=getattr(config.dataset, "train_ratio", 0.70),
            stratify=all_labels,
            random_state=seed,
        )
    )

    val_ratio = getattr(config.dataset, "val_ratio", 0.15)
    test_ratio = getattr(config.dataset, "test_ratio", 0.15)
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    val_data, test_data, val_labels, test_labels, val_ids, test_ids = (
        train_test_split(
            temp_data,
            temp_labels,
            temp_ids,
            train_size=val_fraction_of_temp,
            stratify=temp_labels,
            random_state=seed,
        )
    )

    train_ds = CustomMNISTDataset(
        train_data, train_labels, train_ids, return_sample_ids
    )
    val_ds = CustomMNISTDataset(val_data, val_labels, val_ids, return_sample_ids)
    test_ds = CustomMNISTDataset(
        test_data, test_labels, test_ids, return_sample_ids
    )

    batch_size = getattr(config.dataloader, "batch_size", 64)
    num_workers = getattr(config.dataloader, "num_workers", 2)
    pin_memory = getattr(config.dataloader, "pin_memory", True)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(train_ds, shuffle=train_shuffle, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return (
        train_loader,
        val_loader,
        test_loader,
        train_labels,
        val_labels,
        test_labels,
    )
