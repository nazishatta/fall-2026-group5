"""Regression checks for historical summary comparisons and safe run creation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.v1_mnist.component.pipeline.p3_novelty_detection import (
    run_historical_baseline_verification as verifier,
)


def test_compare_summaries_reports_agreement_and_disagreement() -> None:
    """Compare only specified historical numerical fields."""
    observed = {"baseline.mean": 1.0, "baseline.std": 2.5}
    reference = {"baseline": {"mean": 1.0, "std": 3.0}}
    comparisons = verifier.compare_summaries(observed, reference, atol=1e-9, rtol=1e-8)
    assert comparisons["baseline.mean"]["agrees"] is True
    assert comparisons["baseline.std"]["agrees"] is False
    assert comparisons["baseline.std"]["absolute_difference"] == 0.5


def test_reject_repository_escape() -> None:
    """Never permit configuration values to escape the repository root."""
    with pytest.raises(ValueError, match="outside repository"):
        verifier.resolve_repo_path("../../../../etc/passwd")


def test_existing_run_preflight_blocks_even_without_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing run aborts before loading training data or changing any files."""
    base = tmp_path / "repo"
    base.mkdir()
    (base / "config.yaml").write_text("placeholder", encoding="utf-8")
    output = base / "outputs" / "existing_run"
    output.mkdir(parents=True)
    sentinel = output / "original.json"
    sentinel.write_text("historical", encoding="utf-8")
    monkeypatch.setattr(verifier, "ROOT", base)
    configs = iter([
        SimpleNamespace(benchmark_config="config.yaml", run_id="existing_run", output_root="outputs"),
        SimpleNamespace(),
    ])
    monkeypatch.setattr(verifier, "load_config", lambda _path: next(configs))
    with pytest.raises(FileExistsError, match="must not be overwritten"):
        verifier.run(base / "recheck.yaml", save=True)
    assert sentinel.read_text(encoding="utf-8") == "historical"


def test_nan_comparison_does_not_pass() -> None:
    """Non-finite values cannot be marked as a successful reproduction."""
    result = verifier.compare_summaries(
        {"a": np.nan}, {"a": 1.0}, atol=1e-9, rtol=1e-8
    )
    assert result["a"]["agrees"] is False
