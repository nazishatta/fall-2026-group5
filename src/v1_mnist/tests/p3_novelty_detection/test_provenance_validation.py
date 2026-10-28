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


@pytest.mark.skipif(
    not all(
        (shared_provenance.ROOT / str(path)).is_file()
        for path in shared_provenance.load_manifest()["inputs"].values()
    ),
    reason="Frozen Phase 3 DVC inputs unavailable in this checkout",
)
def test_rejects_incorrect_input_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = shared_provenance.load_manifest()
    monkeypatch.setattr(
        shared_provenance, "sha256_file", lambda path: "0" * 64
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        shared_provenance.verify_shared_inputs(manifest["inputs"])


def test_rejects_noncanonical_svhn_path() -> None:
    """The SVHN file must use the canonical configured location."""
    inputs = {
        "svhn_file": "incorrect/test_32x32.mat",
    }

    with pytest.raises(ValueError, match="Noncanonical"):
        shared_provenance.verify_shared_inputs(inputs)


@pytest.mark.skipif(
    not (
        shared_provenance.ROOT
        / str(shared_provenance.load_manifest()["svhn_file"])
    ).is_file(),
    reason="SVHN DVC dataset unavailable in this checkout",
)
def test_rejects_incorrect_svhn_checksum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing SVHN contents must fail SHA-256 verification."""
    manifest = shared_provenance.load_manifest()
    inputs = {"svhn_file": manifest["svhn_file"]}

    monkeypatch.setattr(
        shared_provenance,
        "sha256_file",
        lambda path: "0" * 64,
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        shared_provenance.verify_shared_inputs(inputs)


def test_rejects_missing_required_input() -> None:
    """Required frozen inputs must never be silently skipped."""
    with pytest.raises(ValueError, match='Missing required'):
        shared_provenance.verify_shared_inputs(
            {'svhn_file': shared_provenance.load_manifest()['svhn_file']},
            required_keys=('checkpoint', 'svhn_file'),
        )
