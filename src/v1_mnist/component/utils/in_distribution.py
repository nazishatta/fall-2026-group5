"""The in-distribution dataset of a Phase 3 or Phase 4 config, and the names that follow from it.

The dataset is read the same way as in Phases 1 and 2: from the ``base.yaml`` in the config's own
folder (``load_config`` merges that file into every config). A config with no ``base.yaml`` next
to it (the historical reorganisation configs, test configs) belongs to MNIST.

Every file name, result key and description that used to say "mnist" is built from the dataset
name here. For MNIST the name *is* ``mnist``, so MNIST runs keep exactly the names and text they
had before this module existed (``qe_mnist_val_scores.npy``, ``"mnist_validation"``, ...); a new
dataset gets its own (``qe_fashion_mnist_val_scores.npy``, ``"fashion_mnist_validation"``, ...).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MNIST = "mnist"

# The word MNIST, not the end of a longer name such as Fashion-MNIST. It may be followed by a
# hyphen ("MNIST-like" becomes "Fashion-MNIST-like").
_BARE_MNIST = re.compile(r"(?<![-\w])MNIST(?!\w)")


@dataclass(frozen=True)
class InDistribution:
    """The dataset a config's network was trained on.

    Attributes:
        name: Registry name, e.g. ``mnist`` or ``fashion_mnist``.
        display_name: Name for text and figure titles, e.g. ``MNIST``.
        class_label: Word for one class, e.g. ``digit`` or ``class``.
    """

    name: str = MNIST
    display_name: str = "MNIST"
    class_label: str = "digit"

    @property
    def is_mnist(self) -> bool:
        """True for MNIST, whose names and text are kept exactly as before."""
        return self.name == MNIST

    @property
    def val_tag(self) -> str:
        """Split tag in score file names: ``mnist_val`` -> ``qe_mnist_val_scores.npy``."""
        return f"{self.name}_val"

    @property
    def validation_key(self) -> str:
        """Result key for the validation split: ``mnist_validation``."""
        return f"{self.name}_validation"

    @property
    def train_key(self) -> str:
        """Result key for the training split: ``mnist_train``."""
        return f"{self.name}_train"

    def key(self, suffix: str) -> str:
        """Any other result key built from the name: ``key("test_read")`` -> ``mnist_test_read``."""
        return f"{self.name}_{suffix}"

    def expected_count(self, expected: Mapping[str, Any], split: str) -> int:
        """``expected["<name>_<split>_count"]`` from a config, e.g. ``mnist_train_count``."""
        key = f"{self.name}_{split}_count"
        if key not in expected:
            raise KeyError(f"config 'expected' has no {key!r}")
        return int(expected[key])

    def wording(self, text: str) -> str:
        """Put this dataset's name into a description written for MNIST.

        MNIST text is returned unchanged. Otherwise "MNIST digit" becomes e.g. "Fashion-MNIST
        class", and every other "MNIST" becomes the display name, also before a hyphen
        ("MNIST-like"); names that end in MNIST (Fashion-MNIST) and lower-case names are left alone.
        """
        if self.is_mnist:
            return text
        text = text.replace("MNIST digits", f"{self.display_name} {self.class_label} images")
        text = text.replace("MNIST digit", f"{self.display_name} {self.class_label}")
        return _BARE_MNIST.sub(self.display_name, text)


def in_distribution(config_path: str | Path) -> InDistribution:
    """The dataset of the config at ``config_path``, read from the ``base.yaml`` next to it."""
    base = Path(config_path).resolve().parent / "base.yaml"
    if not base.is_file():
        return InDistribution()
    raw = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
    dataset = raw.get("dataset") or {}
    name = str(dataset.get("name", MNIST))
    if name == MNIST:
        return InDistribution()
    return InDistribution(
        name=name,
        display_name=str(dataset.get("display_name", name)),
        class_label=str(dataset.get("class_label", "class")),
    )
