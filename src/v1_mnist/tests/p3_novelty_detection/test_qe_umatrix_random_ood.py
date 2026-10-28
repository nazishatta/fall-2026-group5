"""Tests for the Phase 3 QE/U-Matrix and random-score analysis."""

from pathlib import Path

import numpy as np
import pytest

from src.v1_mnist.component.pipeline.p3_novelty_detection.run_qe_umatrix_random_ood import (
    run,
    within_root,
)


def test_repository_path_guard() -> None:
    """Reject a path outside the repository."""
    with pytest.raises(ValueError, match="escapes repository"):
        within_root("../outside")


def test_existing_run_cannot_be_overwritten(tmp_path: Path) -> None:
    """The completed historical run must not be overwritten."""
    config = (
        Path("src/v1_mnist/component/configs/qe_u_matrix_random_ood.yaml")
    )
    with pytest.raises(FileExistsError, match="Run already exists"):
        run(config)


def test_saved_metrics_match_expected_values() -> None:
    """Verify the recorded Phase 3 AUROC and random-score results."""
    import json

    path = Path(
        "outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/"
        "mnist_svhn_qe_umatrix_random_s42_v1/"
        "qe_umatrix_random_metrics.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    paired = data["paired_qe_vs_u_matrix"]
    random_control = data["random_score_null"]

    assert np.isclose(paired["qe_auroc"], 0.9811068135920626)
    assert np.isclose(paired["u_matrix_auroc"], 0.9696289713027191)
    assert np.isclose(paired["qe_minus_u_matrix"], 0.011477842289343565)
    assert paired["difference_ci_95"][0] > 0
    assert 0.49 < random_control["mean_auroc"] < 0.51
    assert data["n_id"] == 10500
    assert data["n_ood"] == 26032
