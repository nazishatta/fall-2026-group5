"""Which image datasets the pipeline can load, and how their images become network input.

A dataset is chosen by ``dataset.name`` in the config (``base.yaml``); the number of channels and
the image size the network sees come from ``dataset.channels`` and ``dataset.image_size``. Images
whose stored size or colour differs are converted (resized, made grey or made colour) on load, so
the same code serves any dataset in ``SOURCES``. To add a dataset, add one entry to ``SOURCES``.

Sample IDs: the original training images come first (IDs 0 .. n_train - 1), then the original test
images. Every step finds an image by this ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Luminance weights (ITU-R BT.601) for turning a colour image grey.
GREY_WEIGHTS = (0.299, 0.587, 0.114)


@dataclass(frozen=True)
class DatasetSource:
    """Where a dataset comes from and how it is stored."""

    torchvision_class: str   # class in torchvision.datasets
    folder: str              # folder torchvision creates under paths.data_root
    native_size: int         # stored height and width
    native_channels: int     # 1 (grey) or 3 (colour)
    idx_files: bool          # stored as the four MNIST-format idx files under <folder>/raw


SOURCES: dict[str, DatasetSource] = {
    "mnist": DatasetSource("MNIST", "MNIST", 28, 1, True),
    "fashion_mnist": DatasetSource("FashionMNIST", "FashionMNIST", 28, 1, True),
    "kmnist": DatasetSource("KMNIST", "KMNIST", 28, 1, True),
    "cifar10": DatasetSource("CIFAR10", "cifar-10-batches-py", 32, 3, False),
}


@dataclass(frozen=True)
class DatasetSpec:
    """The dataset a run uses and the input its network expects (defaults: MNIST)."""

    name: str = "mnist"
    channels: int = 1
    image_size: int = 28
    # Word for one class in output names ("digit" for MNIST, "class" otherwise); not part of identity.
    class_label: str = field(default="digit", compare=False)


def dataset_source(name: str) -> DatasetSource:
    """The registry entry for ``name``; stops with the known names if there is none."""
    try:
        return SOURCES[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name!r}. Known: {', '.join(sorted(SOURCES))} "
            "(add an entry to SOURCES in utils/image_datasets.py)."
        ) from None


def dataset_spec(config: Any) -> DatasetSpec:
    """Read the dataset name, channels and image size from a loaded config."""
    dataset = getattr(config, "dataset", None)
    if dataset is None or getattr(dataset, "name", None) is None:
        raise KeyError("The config has no dataset.name (see configs/base.yaml).")
    source = dataset_source(str(dataset.name))
    return DatasetSpec(
        name=str(dataset.name),
        channels=int(getattr(dataset, "channels", None) or source.native_channels),
        image_size=int(getattr(dataset, "image_size", None) or source.native_size),
        class_label=str(getattr(dataset, "class_label", None) or ("digit" if dataset.name == "mnist" else "class")),
    )


def _as_numpy(values: Any) -> np.ndarray:
    return values.numpy() if hasattr(values, "numpy") else np.asarray(values)


def load_raw_images(data_root: str | Path, name: str, download: bool) -> tuple[np.ndarray, np.ndarray]:
    """All images (uint8, as stored) and labels (int64) of a dataset, in sample-ID order."""
    from torchvision import datasets

    source = dataset_source(name)
    cls = getattr(datasets, source.torchvision_class)
    parts = [cls(root=str(data_root), train=train, download=download) for train in (True, False)]
    images = np.concatenate([_as_numpy(p.data) for p in parts], axis=0)
    labels = np.concatenate([_as_numpy(p.targets) for p in parts], axis=0).astype(np.int64)
    return images, labels


def to_model_input(images: np.ndarray, channels: int, image_size: int) -> np.ndarray:
    """Scale stored uint8 images to 0..1 and give them the channels and size the network expects.

    Grey images that already have the right size are only scaled (N x H x W float32), so MNIST,
    Fashion-MNIST and KMNIST at 28 x 28 are unchanged. Anything else is returned as N x C x H x W
    (N x H x W when ``channels`` is 1): colour is turned grey (or grey repeated to 3 channels) and
    the images are resized with bilinear interpolation and anti-aliasing.
    """
    if channels not in (1, 3):
        raise ValueError(f"dataset.channels must be 1 or 3, not {channels}.")
    scaled = images.astype(np.float32)
    scaled /= 255.0
    if scaled.ndim == 3 and channels == 1 and scaled.shape[1:] == (image_size, image_size):
        return scaled

    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(scaled)
    x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)   # N x C x H x W
    if x.shape[1] == 3 and channels == 1:
        weights = torch.tensor(GREY_WEIGHTS, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x * weights).sum(dim=1, keepdim=True)
    elif x.shape[1] == 1 and channels == 3:
        x = x.repeat(1, 3, 1, 1)
    if tuple(x.shape[2:]) != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear",
                          align_corners=False, antialias=True)
    out = x.numpy().astype(np.float32)
    return out[:, 0] if channels == 1 else out
