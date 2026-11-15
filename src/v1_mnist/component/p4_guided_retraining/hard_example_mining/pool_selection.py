"""The three added-image pools that are compared: SOM, max-softmax (msp) and random.

All three add the same number of images. The random pool also copies the SOM pool's digit mix,
so any difference between them is about which images are chosen, not about which digits.
"""

from __future__ import annotations

import numpy as np


def class_counts(labels: np.ndarray, mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Number of selected images of each digit."""
    return np.bincount(labels[mask], minlength=num_classes)


def msp_pool(confidence: np.ndarray, sample_id: np.ndarray, size: int) -> np.ndarray:
    """The ``size`` images the CNN is least sure about (lowest max-softmax); ties by sample ID."""
    if not 0 < size <= len(confidence):
        raise ValueError(f"Pool size {size} must be between 1 and {len(confidence)}.")
    order = np.lexsort((sample_id, confidence))
    mask = np.zeros(len(confidence), dtype=bool)
    mask[order[:size]] = True
    return mask


def random_pool(labels: np.ndarray, per_class: np.ndarray, seed: int) -> np.ndarray:
    """Random images with ``per_class[c]`` images of digit c, drawn without replacement."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for digit, wanted in enumerate(per_class):
        candidates = np.flatnonzero(labels == digit)
        if wanted > len(candidates):
            raise ValueError(f"Digit {digit}: {wanted} images wanted, only {len(candidates)} exist.")
        mask[rng.choice(candidates, size=int(wanted), replace=False)] = True
    return mask


def resolve_pool(
    kind: str,
    som_pool: np.ndarray,
    msp: np.ndarray,
    hotspot: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    """The pool an arm uses, as a boolean mask over the training images.

    som / msp / hotspot are fixed. ``random`` has the size and digit mix of the SOM pool and
    ``random_hotspot`` those of the hotspot pool; both are redrawn for every seed.
    """
    if kind == "som":
        return som_pool
    if kind == "msp":
        return msp
    if kind == "hotspot":
        return hotspot
    num_classes = int(labels.max()) + 1
    if kind == "random":
        return random_pool(labels, class_counts(labels, som_pool, num_classes), seed)
    if kind == "random_hotspot":
        return random_pool(labels, class_counts(labels, hotspot, num_classes), seed)
    raise ValueError(f"Unknown pool kind '{kind}'.")
