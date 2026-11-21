"""The Phase 3 criteria, judged from six aligned novelty scores in one run.

Two sets of criteria, both fixed before the run that applies them:
  - the group proposal's RQ3 and RQ4 (proposals/Group_proposal.md, success criteria), for every
    dataset: RQ3, the SOM score (S1 QE) detects outside images; RQ4, it is competitive with the best
    baseline, the one of S3 to S6 with the highest AUROC;
  - statements O1 to O4, for a dataset whose own Phase 3 protocol states them (Fashion-MNIST).

The scores (higher = more novel), as numbered in the protocol:
  S1 qe, S2 u_matrix, S3 max_softmax, S4 energy, S5 mahalanobis, S6 knn.
Every score array covers the same in-distribution validation images and the same outside images,
in the same order, so the paired bootstrap compares the scores on identical resamples.

Verdicts (the same two rules for both sets):
  detection (O1 for S1, O4 for S2): met if the AUROC interval lies above 0.5, not met if it lies at
      or below 0.5, inconclusive if it contains 0.5;
  competitive (O2 for S3, O3 for S4, S5, S6): with d = AUROC(S1) - AUROC(other) and its paired
      interval, met if the lower end is above -margin, not met if the upper end is below -margin,
      inconclusive otherwise.
Only existing MNIST functions compute the numbers (evaluate_scores, compare_auroc_paired,
permutation_auroc_null); this module only arranges them and applies the rules above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.ood_comparison import (
    evaluate_scores,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.paired_bootstrap import (
    compare_auroc_paired,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.qe_threshold_sweep import (
    permutation_auroc_null,
)

SCORES = ("qe", "u_matrix", "max_softmax", "energy", "mahalanobis", "knn")   # S1 to S6
BASELINES = ("max_softmax", "energy", "mahalanobis", "knn")                   # S3 to S6: the CNN-based baselines
SCORE_IDS = {name: f"S{index}" for index, name in enumerate(SCORES, start=1)}
MET, NOT_MET, INCONCLUSIVE = "met", "not met", "inconclusive"

ScorePair = tuple[NDArray[np.floating], NDArray[np.floating]]


@dataclass(frozen=True)
class StatementSettings:
    """The protocol's numbers: bootstrap and control sizes and seeds, and the margin."""

    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 42
    paired_iterations: int = 2000
    paired_seed: int = 42
    shuffle_iterations: int = 1000
    shuffle_seed: int = 42
    target_tpr: float = 0.95
    competitive_margin: float = 0.01
    protocol_statements: bool = True     # also judge O1 to O4 (only where the dataset's protocol states them)


def detection_verdict(interval: list[float], chance: float = 0.5) -> str:
    """Met if the whole interval is above chance, not met if at or below it, else inconclusive."""
    low, high = float(interval[0]), float(interval[1])
    if low > chance:
        return MET
    if high <= chance:
        return NOT_MET
    return INCONCLUSIVE


def competitive_verdict(interval: list[float], margin: float) -> str:
    """For d = AUROC(QE) - AUROC(other): met if low > -margin, not met if high < -margin, else inconclusive."""
    low, high = float(interval[0]), float(interval[1])
    if low > -margin:
        return MET
    if high < -margin:
        return NOT_MET
    return INCONCLUSIVE


def check_aligned(scores: dict[str, ScorePair]) -> tuple[int, int]:
    """All six scores present, finite, and of the same two lengths; returns (n_id, n_ood)."""
    missing = [name for name in SCORES if name not in scores]
    if missing:
        raise ValueError(f"Missing scores: {missing}")
    sizes = {(len(np.ravel(i)), len(np.ravel(o))) for i, o in (scores[name] for name in SCORES)}
    if len(sizes) != 1:
        raise ValueError(f"Score arrays are not aligned: {sorted(sizes)}")
    for name in SCORES:
        if not all(np.isfinite(np.asarray(part, dtype=np.float64)).all() for part in scores[name]):
            raise ValueError(f"{name}: scores must be finite")
    return sizes.pop()


def judge(scores: dict[str, ScorePair], settings: StatementSettings) -> dict[str, Any]:
    """Per-score measures, QE against every other score, the label-shuffle control, and O1 to O4."""
    n_id, n_ood = check_aligned(scores)
    per_score = {}
    for name in SCORES:
        result = evaluate_scores(*scores[name], bootstrap_repeats=settings.bootstrap_iterations,
                                 seed=settings.bootstrap_seed, target_tpr=settings.target_tpr)
        per_score[name] = {"score_id": SCORE_IDS[name], **result, "aupr_chance_level": n_ood / (n_id + n_ood)}

    paired = {}
    for name in SCORES[1:]:
        result = compare_auroc_paired(*scores["qe"], *scores[name],
                                      iterations=settings.paired_iterations, seed=settings.paired_seed)
        paired[name] = {
            "comparison": f"S1 qe minus {SCORE_IDS[name]} {name}",
            "qe_auroc": result["qe_auroc"],
            "other_auroc": result["msp_auroc"],          # compare_auroc_paired names the second score MSP
            "qe_minus_other_auroc": result["qe_minus_msp_auroc"],
            "difference_ci_95": result["difference_ci_95"],
            "bootstrap_iterations": result["bootstrap_iterations"],
            "seed": result["seed"],
        }

    control = permutation_auroc_null(*scores["qe"], iterations=settings.shuffle_iterations,
                                     seed=settings.shuffle_seed)

    def auroc_interval(name: str) -> list[float]:
        return per_score[name]["confidence_intervals_95"]["auroc"]

    margin = settings.competitive_margin
    best = max(BASELINES, key=lambda name: per_score[name]["auroc"])
    proposal = {
        "RQ3": {"criterion": "the SOM score (S1 QE) detects outside images: AUROC interval excludes 0.5",
                "interval": auroc_interval("qe"), "verdict": detection_verdict(auroc_interval("qe"))},
        "RQ4": {"criterion": (f"competitive with the best baseline: lower end of the paired interval of S1 minus "
                              f"the best of S3-S6 above -{margin}"),
                "best_baseline": best, "best_baseline_auroc": per_score[best]["auroc"],
                "interval": paired[best]["difference_ci_95"],
                "verdict": competitive_verdict(paired[best]["difference_ci_95"], margin),
                "note": ("The best baseline is picked by its AUROC on these images, as the criterion says. The "
                         "classical baselines (Isolation Forest, One-Class SVM, autoencoder) are compared in "
                         "their own steps; if one of them has a higher AUROC, RQ4 is checked against it there.")},
    }
    statements = {
        "O1": {"statement": "the SOM (S1 QE) detects outside images", "primary": False,
               "interval": auroc_interval("qe"), "verdict": detection_verdict(auroc_interval("qe"))},
        "O2": {"statement": "the SOM is competitive with max-softmax (S1 minus S3)", "primary": True,
               "interval": paired["max_softmax"]["difference_ci_95"],
               "verdict": competitive_verdict(paired["max_softmax"]["difference_ci_95"], margin)},
        "O3": {"statement": "the SOM is competitive with energy, Mahalanobis and nearest neighbours",
               "primary": False,
               "per_score": {name: {"interval": paired[name]["difference_ci_95"],
                                    "verdict": competitive_verdict(paired[name]["difference_ci_95"], margin)}
                             for name in ("energy", "mahalanobis", "knn")}},
        "O4": {"statement": "the U-Matrix (S2) detects outside images", "primary": False,
               "interval": auroc_interval("u_matrix"), "verdict": detection_verdict(auroc_interval("u_matrix"))},
    }
    return {
        "n_id": n_id, "n_ood": n_ood, "aupr_chance_level": n_ood / (n_id + n_ood),
        "settings": vars(settings), "scores": per_score, "paired_against_qe": paired,
        "label_shuffle_control": control, "proposal_criteria": proposal,
        **({"statements": statements} if settings.protocol_statements else {}),
        "rules": {
            "detection": "met if the AUROC interval lies above 0.5; not met if at or below 0.5; else inconclusive",
            "competitive": (f"d = AUROC(S1) - AUROC(other); met if the lower end of d's paired interval is above "
                            f"-{margin}; not met if its upper end is below -{margin}; else inconclusive"),
        },
    }
