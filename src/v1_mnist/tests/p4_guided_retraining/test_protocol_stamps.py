"""Both stamped Phase 4 protocols and their retraining.yaml files still match their stamps.

A stamp proves the rules existed, unchanged, before the test split was read. Results are recorded in
separate files, so the stamped files must stay byte-identical for that proof to remain checkable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
STAMPS = {
    "mnist": (ROOT / "src/v1_mnist/docs/p4_guided_retraining",
              ROOT / "src/v1_mnist/component/configs/retraining.yaml"),
    "fashion_mnist": (ROOT / "src/v1_fashion_mnist/docs/p4_guided_retraining",
                      ROOT / "src/v1_fashion_mnist/component/configs/retraining.yaml"),
}


def _stamp(folder: Path) -> dict[str, str]:
    lines = (folder / "retraining_protocol_stamp.txt").read_text(encoding="utf-8").splitlines()
    return dict(line.split(": ", 1) for line in lines if ": " in line)


@pytest.mark.parametrize("dataset", sorted(STAMPS))
def test_the_stamped_protocol_and_config_are_unchanged(dataset: str) -> None:
    folder, config = STAMPS[dataset]
    stamp = _stamp(folder)
    protocol = folder / stamp["protocol"]
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == stamp["protocol_sha256"], protocol
    assert hashlib.sha256(config.read_bytes()).hexdigest() == stamp["config_sha256"], config
