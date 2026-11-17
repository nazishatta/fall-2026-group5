"""Selection and comparison rules for the Optuna SOM tuning (labels never used).

Rules fixed in OPTUNA_TUNING_PROTOCOL.md before any real run.

Selection (one setting per grid):
1. Keep the Pareto-optimal trials on (validation QE, validation TE1), both minimised.
2. Among them, pick the lowest equal-weight rank sum of validation QE, TE1 and TE1+2
   (the original grid-selection protocol's rule, used within one grid only).
3. Ties: smaller init_neighborhood, then fewer epochs, then the earlier trial.

Confirmation (Optuna choice vs the 3-point search choice, on new seeds):
per metric, the paired difference (Optuna minus 3-point) over seeds with a t-interval;
"better" / "worse" only when the interval excludes 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TrialResult:
    """One finished trial: its settings and validation metrics."""

    number: int
    init_neighborhood: int
    epochs: int
    val_qe: float
    val_te1_pct: float
    val_te12_pct: float


def parse_grid(grid: str) -> tuple[int, int]:
    """'15x15' -> (15, 15)."""
    parts = grid.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise ValueError(f"Grid must look like '15x15', got {grid!r}.")
    return int(parts[0]), int(parts[1])


def build_run_id(grid: str, n_trials: int, seed: int, revision: str, smoke: bool = False) -> str:
    """Run ID that names the grid, trial budget, seed and revision (never a week number)."""
    run_id = f"som_optuna_{grid}_t{n_trials}_s{seed}_{revision}"
    return f"{run_id}_smoke" if smoke else run_id


def _dominates(a: TrialResult, b: TrialResult) -> bool:
    """True when a is no worse than b on QE and TE1 and better on at least one."""
    no_worse = a.val_qe <= b.val_qe and a.val_te1_pct <= b.val_te1_pct
    better = a.val_qe < b.val_qe or a.val_te1_pct < b.val_te1_pct
    return no_worse and better


def pareto_front(trials: Sequence[TrialResult]) -> list[TrialResult]:
    """Trials that no other trial dominates on (validation QE, validation TE1)."""
    return [t for t in trials if not any(_dominates(o, t) for o in trials if o is not t)]


def average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks starting at 1 (smallest value first); tied values share their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def select_trial(trials: Sequence[TrialResult]) -> TrialResult:
    """Apply the protocol's rule to one grid's finished trials."""
    if not trials:
        raise ValueError("No finished trials to choose from.")
    front = pareto_front(trials)
    qe = average_ranks([t.val_qe for t in front])
    te1 = average_ranks([t.val_te1_pct for t in front])
    te12 = average_ranks([t.val_te12_pct for t in front])
    scored = [
        (qe[i] + te1[i] + te12[i], t.init_neighborhood, t.epochs, t.number, t)
        for i, t in enumerate(front)
    ]
    return min(scored, key=lambda row: row[:4])[4]


def paired_difference(diffs: Sequence[float], level: float = 0.95) -> dict[str, float]:
    """Mean of paired differences and its two-sided t-interval (n - 1 degrees of freedom)."""
    d = np.asarray(diffs, dtype=float)
    if d.size < 2:
        raise ValueError("Need at least 2 paired differences.")
    mean = float(d.mean())
    half = float(stats.t.ppf(0.5 + level / 2, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size))
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half, "n": int(d.size)}


def metric_verdict(ci_low: float, ci_high: float) -> str:
    """For a lower-is-better metric and the difference (Optuna minus 3-point)."""
    if ci_high < 0:
        return "optuna better"
    if ci_low > 0:
        return "optuna worse"
    return "no clear difference"


def overall_verdict(verdicts: Sequence[str]) -> str:
    """Combine the per-metric verdicts (QE, TE1, TE1+2) into one statement."""
    better = "optuna better" in verdicts
    worse = "optuna worse" in verdicts
    if better and worse:
        return "trade-off: the two settings balance QE and TE differently"
    if better:
        return "Optuna setting better"
    if worse:
        return "3-point setting better"
    return "3-point search confirmed: no clear difference"
