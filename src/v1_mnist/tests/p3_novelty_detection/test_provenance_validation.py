"""Safety tests for frozen Phase 3 provenance."""

from __future__ import annotations

import pytest

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark import (
    shared_provenance,
)


def test_rejects_noncanonical_input_path() -> None:
    manifest = shared_provenance.load_manifest()
    inputs = dict(manifest["inputs"])
    inputs["checkpoint"] = "incorrect/checkpoint.pth"

    with pytest.raises(ValueError, match="Noncanonical"):
        shared_provenance.verify_shared_inputs(inputs)


def test_rejects_incorrect_input_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = shared_provenance.load_manifest()
    monkeypatch.setattr(
        shared_provenance, "sha256_file", lambda path: "0" * 64
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        shared_provenance.verify_shared_inputs(manifest["inputs"])
