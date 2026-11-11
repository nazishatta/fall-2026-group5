"""Every figure in the Phase 3 report is the PDF its run produced."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src.v1_mnist.component.pipeline.p3_novelty_detection.copy_report_figures import (
    CONFIG_PATH,
    copy_figures,
    load_figures,
    recorded_sha256,
)

ROOT = Path(__file__).resolve().parents[4]
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
REPORT_DIR = ROOT / CONFIG["report_dir"]
NOT_RUN_OUTPUTS = {
    "fig/gwu_logo.pdf",
    "fig/novelty_framework/phase3_novelty_framework.pdf",
}


def test_every_report_figure_is_listed() -> None:
    """Each figure the report includes is either listed with its run or is the logo/diagram."""
    tex = (REPORT_DIR / CONFIG["report_tex"]).read_text(encoding="utf-8")
    included = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    listed = {entry["report_file"] for entry in CONFIG["figures"]}
    assert included - NOT_RUN_OUTPUTS == listed


def test_every_source_run_folder_is_in_the_repository() -> None:
    """Each figure points at a run folder that exists and has a manifest row."""
    manifest = (
        ROOT / "outputs/v1_mnist/p3_novelty_detection/run_manifest.csv"
    ).read_text(encoding="utf-8")
    for figure in load_figures():
        assert figure["run_folder"].is_dir(), figure["run_folder"]
        assert f",{figure['run_folder'].name}," in manifest


def test_report_copies_match_their_runs() -> None:
    """Report copies exist and, where the run recorded a fingerprint, match it."""
    assert copy_figures(load_figures(), check_only=True) == []


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
