"""Validate Phase 3 frozen input provenance against the shared manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.v1_mnist.component.utils.som_io import sha256_file

ROOT = Path(__file__).resolve().parents[5]
MANIFEST = ROOT / "src/v1_mnist/component/configs/phase3_shared_provenance.yaml"


def load_manifest() -> dict[str, Any]:
    """Read canonical Phase 3 frozen input references."""
    with MANIFEST.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def verify_shared_inputs(
    inputs: dict[str, str],
    required_keys: tuple[str, ...] = (),
) -> None:
    """Verify canonical frozen inputs and fail on missing required keys."""
    shared = load_manifest()
    missing = set(required_keys) - set(inputs)
    if missing:
        raise ValueError(f"Missing required Phase 3 inputs: {sorted(missing)}")
    unknown = set(required_keys) - set(shared['expected_sha256'])
    if unknown:
        raise ValueError(f"Unknown provenance keys: {sorted(unknown)}")
    for key, expected_hash in shared["expected_sha256"].items():
        if key not in inputs:
            continue
        path = ROOT / inputs[key]
        canonical_path = (
            shared["svhn_file"] if key == "svhn_file"
            else shared["inputs"][key]
        )
        if inputs[key] != canonical_path:
            raise ValueError(f"Noncanonical Phase 3 input path: {key}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"Frozen Phase 3 SHA-256 mismatch: {key}")
