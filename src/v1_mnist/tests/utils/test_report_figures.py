"""Tests for the report-figure copy utility (component/utils/report_figures.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.v1_mnist.component.utils.report_figures import copy_figures, recorded_sha256


def test_mismatched_run_figure_is_refused(tmp_path: Path) -> None:
    """A run figure whose fingerprint differs from the recorded one is not copied."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "figure.pdf").write_bytes(b"changed")
    (run / "figure_metrics.json").write_text(
        '{"figure_sha256": "' + "0" * 64 + '"}', encoding="utf-8"
    )
    figures = [
        {
            "report_file": tmp_path / "report" / "figure.pdf",
            "run_folder": run,
            "run_file": run / "figure.pdf",
            "hash_record": run / "figure_metrics.json",
        }
    ]
    problems = copy_figures(figures)
    assert problems and "differs" in problems[0]
    assert not (tmp_path / "report" / "figure.pdf").exists()
    with pytest.raises(ValueError):
        (run / "figure_metrics.json").write_text("{}", encoding="utf-8")
        recorded_sha256(run / "figure_metrics.json")
