"""Label-preserving image changes for targeted augmentation (small affine moves only).

Each altered copy gets a random rotation, shift, zoom and shear drawn from the ranges in
``configs/retraining.yaml``. No flips and no large rotations: a flipped or turned-over digit can
become another digit (6 and 9), which would change the label. Plain NumPy/SciPy, seeded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform


@dataclass(frozen=True)
class TransformRanges:
    """Largest size of each change (all symmetric around zero, zoom around one)."""

    max_rotation_degrees: float
    max_shift_pixels: float
    min_zoom: float
    max_zoom: float
    max_shear_degrees: float


def random_affine(rng: np.random.Generator, ranges: TransformRanges) -> np.ndarray:
    """One random 2x2 matrix (rotation, shear and zoom) and a shift, as (matrix, shift)."""
    angle = np.deg2rad(rng.uniform(-ranges.max_rotation_degrees, ranges.max_rotation_degrees))
    shear = np.deg2rad(rng.uniform(-ranges.max_shear_degrees, ranges.max_shear_degrees))
    zoom = rng.uniform(ranges.min_zoom, ranges.max_zoom)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shear_matrix = np.array([[1.0, np.tan(shear)], [0.0, 1.0]])
    matrix = rotation @ shear_matrix / zoom
    shift = rng.uniform(-ranges.max_shift_pixels, ranges.max_shift_pixels, size=2)
    return matrix, shift


def alter_image(image: np.ndarray, rng: np.random.Generator, ranges: TransformRanges) -> np.ndarray:
    """One altered copy of a 2-D image (0..1), turned about the image centre."""
    matrix, shift = random_affine(rng, ranges)
    centre = (np.array(image.shape, dtype=float) - 1.0) / 2.0
    offset = centre - matrix @ centre + shift
    out = affine_transform(image, matrix, offset=offset, order=1, mode="constant", cval=0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_copies(
    images: np.ndarray, copies: int, seed: int, ranges: TransformRanges
) -> np.ndarray:
    """``copies`` altered versions of every image: shape (len(images) * copies, 28, 28).

    Order: all images' first copy, then all images' second copy, and so on.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((len(images) * copies, *images.shape[1:]), dtype=np.float32)
    row = 0
    for _ in range(copies):
        for image in images:
            out[row] = alter_image(image, rng, ranges)
            row += 1
    return out
