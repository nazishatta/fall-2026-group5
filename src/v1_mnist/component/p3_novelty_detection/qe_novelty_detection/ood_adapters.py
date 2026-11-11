"""Dataset-specific image adapters; imported lazily to keep numeric tests lightweight."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
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


def svhn_test_loader(
    svhn_file: str | Path,
    image_size: Sequence[int],
    batch_size: int,
    transform: Any | None = None,
) -> Any:
    """Return an unshuffled loader over the SVHN test split.

    Every Phase 3 step reads SVHN the same way: the test split stored next to the
    shared-provenance ``test_32x32.mat``, never downloaded, in file order.

    Args:
        svhn_file: Path of ``test_32x32.mat`` from the shared provenance manifest.
        image_size: Height and width for the preregistered SVHN transform.
        batch_size: Loader batch size.
        transform: Optional transform used instead of the preregistered one.

    Returns:
        A ``torch.utils.data.DataLoader`` with ``shuffle=False``.
    """
    from torch.utils.data import DataLoader
    from torchvision.datasets import SVHN

    size = (int(image_size[0]), int(image_size[1]))
    dataset = SVHN(
        root=str(Path(svhn_file).parent),
        split="test",
        download=False,
        transform=transform if transform is not None else image_transform("svhn", size),
    )
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
