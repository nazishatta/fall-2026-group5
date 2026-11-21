"""Validate Phase 3 frozen input provenance against the shared manifest.

Each dataset has its own manifest, ``phase3_shared_provenance.yaml`` in its config folder (the
folder that holds its ``base.yaml``). A config is checked against the manifest in its own folder;
a config in a folder without ``base.yaml`` (historical reorganisation configs, tests) and a call
that names no config use the MNIST manifest, as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.v1_mnist.component.utils.som_io import sha256_file

ROOT = Path(__file__).resolve().parents[4]
MANIFEST_NAME = "phase3_shared_provenance.yaml"
MANIFEST = ROOT / "src/v1_mnist/component/configs" / MANIFEST_NAME


def manifest_path(config_path: str | Path | None = None) -> Path:
    """The manifest for a config: the one in its own dataset folder, else the MNIST one."""
    if config_path is None:
        return MANIFEST
    path = Path(config_path)
    folder = (path if path.is_absolute() else ROOT / path).resolve().parent
    if not (folder / "base.yaml").is_file():
        return MANIFEST
    manifest = folder / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{manifest} is missing: every dataset's config folder needs its own Phase 3 manifest"
        )
    return manifest


def load_manifest(config_path: str | Path | None = None) -> dict[str, Any]:
    """Read canonical Phase 3 frozen input references for the config's dataset."""
    with manifest_path(config_path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def verify_shared_inputs(
    inputs: dict[str, str],
    required_keys: tuple[str, ...] = (),
    config_path: str | Path | None = None,
) -> None:
    """Verify canonical frozen inputs and fail on missing required keys."""
    shared = load_manifest(config_path)
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
