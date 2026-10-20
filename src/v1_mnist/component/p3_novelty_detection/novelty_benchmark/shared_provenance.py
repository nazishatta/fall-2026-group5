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


def verify_shared_inputs(inputs: dict[str, str]) -> None:
    """Fail if configured frozen inputs differ from shared provenance."""
    shared = load_manifest()
    for key, expected_hash in shared["expected_sha256"].items():
        if key not in inputs:
            continue
        path = ROOT / inputs[key]
        if inputs[key] != shared["inputs"][key]:
            raise ValueError(f"Noncanonical Phase 3 input path: {key}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"Frozen Phase 3 SHA-256 mismatch: {key}")
    if "svhn_file" in inputs and inputs["svhn_file"] != shared["svhn_file"]:
        raise ValueError("Noncanonical Phase 3 SVHN path")
