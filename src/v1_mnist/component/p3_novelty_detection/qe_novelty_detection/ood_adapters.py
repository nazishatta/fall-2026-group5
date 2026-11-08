"""Dataset-specific image adapters; imported lazily to keep numeric tests lightweight."""

from __future__ import annotations

from typing import Any


def image_transform(dataset: str, image_size: tuple[int, int]) -> Any:
    """Return the preregistered image-to-tensor transformation."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if len(image_size) != 2 or any(value <= 0 for value in image_size):
        raise ValueError("Image size must be a positive height/width pair")
    if dataset == "svhn":
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(
                    image_size, interpolation=InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.ToTensor(),
            ]
        )
    raise ValueError(f"Unsupported OOD dataset: {dataset}")
