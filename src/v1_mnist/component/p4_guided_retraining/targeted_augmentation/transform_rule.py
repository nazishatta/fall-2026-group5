"""The transform-check rule: how small the augmentation ranges must be, and what happens if they are not.

Config, ``augmentation.transform_check`` (both settings optional):
  max_accuracy_drop  the Phase 1 network's validation accuracy on altered images may be at most this much
                     lower than on the originals (0.02 = 2 percentage points). Without it nothing is
                     enforced (MNIST, whose check was judged by hand before this rule existed).
  max_halvings       if the configured ranges fail, halve all four and check again, at most this many
                     times (rotation, shift and shear halved; zoom halved around 1). The first ranges that
                     pass are used for training. Without it (or 0) a failed check simply stops the run.

Fashion-MNIST applied this rule by hand (revisions r2 and r3, docs/DEVIATIONS.md, 2026-11-21) before it
was in the code; later datasets set ``max_halvings: 2`` and no decision is taken by hand.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.v1_mnist.component.utils.retraining_settings import ExperimentSettings

ZOOM_KEYS = ("min_zoom", "max_zoom")


def check_path(settings: ExperimentSettings) -> Path:
    """Where the transform check writes its result."""
    return settings.output_dir / "check" / "augmentation_check.json"


def _rule(settings: ExperimentSettings) -> dict[str, Any]:
    return settings.options.get("transform_check") or {}


def accuracy_drop_limit(settings: ExperimentSettings) -> float | None:
    """The largest allowed accuracy drop on altered images, or None when the config sets none."""
    limit = _rule(settings).get("max_accuracy_drop")
    return None if limit is None else float(limit)


def max_halvings(settings: ExperimentSettings) -> int:
    """How many times the ranges may be halved after a failed check (0: none)."""
    return int(_rule(settings).get("max_halvings") or 0)


def halved(transforms: dict[str, float], times: int) -> dict[str, float]:
    """The ranges halved ``times`` times: zoom around 1, everything else around 0."""
    factor = 0.5 ** times
    return {key: (1.0 + (float(value) - 1.0) * factor if key in ZOOM_KEYS else float(value) * factor)
            for key, value in transforms.items()}


def attempts_until_pass(
    transforms: dict[str, float], limit: float | None, halvings: int, drop_of: Callable[[dict[str, float]], float],
) -> list[dict[str, Any]]:
    """Check the ranges, then halve them, until one passes or ``halvings`` halvings have failed.

    ``drop_of`` returns the accuracy drop for some ranges. Without a limit only the configured ranges
    are measured. Every attempt is returned; the last one is the result.
    """
    attempts: list[dict[str, Any]] = []
    for times in range(halvings + 1 if limit is not None else 1):
        ranges = halved(transforms, times)
        drop = float(drop_of(ranges))
        attempts.append({"halvings": times, "transforms": ranges, "accuracy_drop": drop,
                         "passed": None if limit is None else bool(drop <= limit)})
        if limit is None or drop <= limit:
            break
    return attempts


def accepted_transforms(settings: ExperimentSettings) -> dict[str, Any]:
    """The ranges training uses: the config's, or the ones the check accepted when halving is allowed."""
    if max_halvings(settings) == 0:
        return dict(settings.options["transforms"])
    path = check_path(settings)
    if not path.is_file():
        raise RuntimeError(f"{path} is missing: run --stage check first (it decides the ranges).")
    return dict(json.loads(path.read_text(encoding="utf-8"))["transforms"])
