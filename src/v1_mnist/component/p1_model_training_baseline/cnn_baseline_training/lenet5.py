from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """LeNet-5 baseline architecture.

    The input channels and image size come from the config (dataset.channels, dataset.image_size);
    the defaults are MNIST's 1 x 28 x 28, which gives the original 16 x 5 x 5 flattened size.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28) -> None:
        super(LeNet5, self).__init__()
        side = ((image_size // 2) - 4) // 2   # after conv1 (same size), pool, conv2 (5x5), pool
        if side < 1:
            raise ValueError(f"image_size {image_size} is too small for LeNet-5 (at least 12).")
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Penultimate feature layer info
        self.feature_layer_name: str = "fc2"
        self.feature_dim: int = 84

        self.fc1 = nn.Linear(16 * side * side, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_feature_layer_name(self) -> str:
        return self.feature_layer_name
