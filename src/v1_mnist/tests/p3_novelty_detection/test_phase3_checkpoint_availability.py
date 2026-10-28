"""Guard Phase 3 checkpoint checks in DVC-incomplete clones."""

from pathlib import Path

import pytest

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.shared_provenance import (
    ROOT,
    load_manifest,
    verify_shared_inputs,
)


def checkpoint_path() -> Path:
    """Return the checkpoint path from the canonical manifest."""
    manifest = load_manifest()
    return ROOT / str(manifest["inputs"]["checkpoint"])


@pytest.mark.skipif(
    not checkpoint_path().is_file(),
    reason="Frozen CNN checkpoint unavailable; restore DVC inputs",
)
def test_frozen_checkpoint_provenance() -> None:
    """Verify the available checkpoint against its frozen SHA-256."""
    relative = str(load_manifest()["inputs"]["checkpoint"])
    verify_shared_inputs(
        {"checkpoint": relative},
        required_keys=("checkpoint",),
    )
