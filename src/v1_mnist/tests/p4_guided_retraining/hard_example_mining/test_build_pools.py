"""Hard example mining: the run-ID preflight and the settings read from retraining.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.v1_mnist.component.utils.run_provenance import ensure_not_done, write_json

CONFIG = Path(__file__).resolve().parents[3] / "component" / "configs" / "retraining.yaml"


def test_finished_run_is_never_overwritten(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    ensure_not_done(results)  # nothing there yet: fine
    write_json(results, {"run_id": "x"})
    with pytest.raises(FileExistsError):
        ensure_not_done(results)


def test_pool_rules_are_in_the_config_and_unchanged() -> None:
    """Fixed in RETRAINING_PROTOCOL.md section 3 before any retraining result."""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert raw["pools"] == {"purity_cutoff": 0.95, "qe_percentile": 95}
    assert raw["run"]["revision"] == ""
