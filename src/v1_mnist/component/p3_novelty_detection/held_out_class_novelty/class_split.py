"""Known and unknown digits of one split (protocol sections 2 and 3). NumPy only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.v1_mnist.component.utils.frozen_split import SplitImages


@dataclass(frozen=True)
class ClassSplit:
    """The images of one split: training (known digits), normal (known) and novel (unknown) validation images.

    ``*_y`` are the known digits renumbered 0..K-1 in increasing digit order (what the network learns);
    ``novel_digits`` keeps the original digits of the novel images, for reporting only.
    """

    known: tuple[int, ...]
    unknown: tuple[int, ...]
    train_x: np.ndarray
    train_y: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    novel_x: np.ndarray
    novel_digits: np.ndarray
    train_ids: np.ndarray
    normal_ids: np.ndarray
    novel_ids: np.ndarray


def known_digits(unknown: tuple[int, ...], num_classes: int) -> tuple[int, ...]:
    """Every digit that is not unknown, in increasing order."""
    if len(set(unknown)) != len(unknown) or not all(0 <= d < num_classes for d in unknown):
        raise ValueError(f"Unknown digits must be distinct and inside 0..{num_classes - 1}: {unknown}")
    known = tuple(d for d in range(num_classes) if d not in unknown)
    if len(known) < 2:
        raise ValueError("At least two known digits are needed.")
    return known


def renumber(labels: np.ndarray, known: tuple[int, ...]) -> np.ndarray:
    """Map known digits to 0..K-1 (increasing digit order). Fails on an unknown digit."""
    lookup = np.full(max(known) + 1, -1, dtype=np.int64)
    lookup[list(known)] = np.arange(len(known))
    labels = np.asarray(labels, dtype=np.int64)
    out = lookup[np.clip(labels, 0, len(lookup) - 1)]
    if (labels < 0).any() or (labels > max(known)).any() or (out < 0).any():
        raise ValueError("renumber() got a digit that is not known.")
    return out


def make_split(unknown: tuple[int, ...], num_classes: int, train: SplitImages, val: SplitImages) -> ClassSplit:
    """Select the images of one split from the frozen training and validation splits."""
    known = known_digits(unknown, num_classes)
    tr = np.isin(train.labels, known)
    va_known = np.isin(val.labels, known)
    va_unknown = np.isin(val.labels, unknown)
    if not (va_known ^ va_unknown).all():
        raise ValueError("Every validation image must be either known or unknown.")
    return ClassSplit(
        known=known, unknown=tuple(unknown),
        train_x=train.images[tr], train_y=renumber(train.labels[tr], known),
        normal_x=val.images[va_known], normal_y=renumber(val.labels[va_known], known),
        novel_x=val.images[va_unknown], novel_digits=val.labels[va_unknown].astype(np.int64),
        train_ids=train.sample_ids[tr], normal_ids=val.sample_ids[va_known], novel_ids=val.sample_ids[va_unknown],
    )


def cells_covering(bmu: np.ndarray, share: float) -> int:
    """Fewest map cells that together hold at least ``share`` of the images (most-populated cells first)."""
    if not 0 < share <= 1 or len(bmu) == 0:
        raise ValueError("share must be in (0, 1] and there must be images.")
    counts = np.sort(np.bincount(np.asarray(bmu, dtype=np.int64)))[::-1]
    return int(np.searchsorted(np.cumsum(counts), share * len(bmu) - 1e-9) + 1)
