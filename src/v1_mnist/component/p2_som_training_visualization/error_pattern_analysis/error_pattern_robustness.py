"""Error-pattern (error-geography) statistics for one SOM, judged against "errors placed at random".

Rules: docs/p2_som_training_visualization/error_pattern_analysis/ERROR_PATTERN_ROBUSTNESS_CRITERIA.md. Nothing here is specific to MNIST,
the grid size or the CNN: every claim is compared with the same chance baseline,
obtained by shuffling which validation images are errors (keeping each image's
neuron and class), so the same code and rules apply to any dataset.

Hotspot concentration (error hotspots, held out):
  - hotspots = top k occupied neurons by 95% Wilson lower bound of error rate,
    k = max(5, round(5% of neurons)); chosen on one half of the validation set.
  - ratio = hotspots' share of the other half's errors / their share of its images.
  - statistic = median ratio over the configured stratified splits x 2 directions
    (analysis.robustness.split_repeats).
  - chance baseline: the same statistic after shuffling error labels
    (analysis.robustness.shuffles times).
Overlap regions:
  - Spearman of neuron error rate with class purity (expect negative) and with
    normalized class entropy (expect positive), over occupied neurons,
    each compared with the same shuffled-label baseline.
Secondary (novelty signal, reported only): BMU distance, errors vs correct.

Usage (from the code root):
  python src/v1_mnist/component/p2_som_training_visualization/error_pattern_analysis/error_pattern_robustness.py [--config CONFIG]
Writes outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/robustness/<selected_model>_error_pattern_robustness.json.
"""

from __future__ import annotations

from types import SimpleNamespace

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.som_io import load_selected_som, load_split, reconstruct_assignments
from src.v1_mnist.component.utils.stats import normalized_entropy
from src.v1_mnist.component.utils.hotspots import n_hotspots, top_hotspots
from src.v1_mnist.component.utils.run_settings import (
    DEFAULT_CONFIG,
    analysis_config,
    load_analysis_settings,
    robustness_dir,
    verify_selected_model,
)
from src.v1_mnist.component.utils.logging import get_logger

logger = get_logger("v1_mnist.analysis.error_pattern_robustness")

@dataclass(frozen=True)
class RobustnessSettings:
    """Settings of the robustness check (configs: analysis.robustness and the hotspot rule).

    Fixed in advance in ERROR_PATTERN_ROBUSTNESS_CRITERIA.md.
    """

    n_splits: int
    n_permutations: int
    min_errors: int
    split_seed_base: int
    perm_seed_base: int
    hotspot_fraction: float
    min_hotspots: int


def robustness_settings(analysis: SimpleNamespace) -> RobustnessSettings:
    """Read the robustness settings from a config's analysis section."""
    rob = analysis.robustness
    return RobustnessSettings(
        n_splits=int(rob.split_repeats),
        n_permutations=int(rob.shuffles),
        min_errors=int(rob.min_errors),
        split_seed_base=int(rob.split_seed_base),
        perm_seed_base=int(rob.shuffle_seed_base),
        hotspot_fraction=float(analysis.hotspot_fraction),
        min_hotspots=int(analysis.min_hotspots),
    )


def stratified_halves(err: np.ndarray, rep: int, split_seed_base: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(split_seed_base + rep)
    a, b = [], []
    for value in (True, False):
        idx = rng.permutation(np.flatnonzero(err == value))
        h = len(idx) // 2
        a.append(idx[:h]); b.append(idx[h:])
    return np.concatenate(a), np.concatenate(b)


def held_out_ratios(bmu: np.ndarray, err: np.ndarray, num_neurons: int, k: int,
                    cfg: RobustnessSettings) -> np.ndarray:
    ratios = []
    for rep in range(cfg.n_splits):
        a, b = stratified_halves(err, rep, cfg.split_seed_base)
        for fit, test in ((a, b), (b, a)):
            hot = top_hotspots(bmu[fit], err[fit], num_neurons, k)
            in_hot = np.isin(bmu[test], hot)
            img_share = in_hot.mean()
            err_share = err[test][in_hot].sum() / err[test].sum()
            ratios.append(err_share / img_share if img_share > 0 else np.nan)
    r = np.asarray(ratios)
    return r[np.isfinite(r)]


def overlap_correlations(bmu: np.ndarray, err: np.ndarray, purity: np.ndarray,
                         entropy: np.ndarray, occ: np.ndarray, support: np.ndarray,
                         num_neurons: int) -> tuple[float, float]:
    rate = np.bincount(bmu, weights=err, minlength=num_neurons)[occ] / support[occ]
    return (float(spearmanr(rate, purity).statistic), float(spearmanr(rate, entropy).statistic))


def p_upper(observed: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def p_lower(observed: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null <= observed)) / (len(null) + 1))


def analyse(bmu: np.ndarray, err_bool: np.ndarray, labels: np.ndarray, num_neurons: int, num_classes: int,
            cfg: RobustnessSettings) -> dict:
    err = err_bool.astype(float)
    k = n_hotspots(num_neurons, cfg.hotspot_fraction, cfg.min_hotspots)
    support = np.bincount(bmu, minlength=num_neurons)
    occ = support > 0
    counts = np.zeros((num_neurons, num_classes))
    np.add.at(counts, (bmu, labels.astype(int)), 1)
    purity = counts[occ].max(1) / support[occ]
    entropy = np.array([normalized_entropy(c) for c in counts[occ]])

    ratios = held_out_ratios(bmu, err, num_neurons, k, cfg)
    obs_ratio = float(np.median(ratios))
    obs_p, obs_e = overlap_correlations(bmu, err, purity, entropy, occ, support, num_neurons)

    null_ratio, null_p, null_e = [], [], []
    for i in range(cfg.n_permutations):
        shuffled = np.random.default_rng(cfg.perm_seed_base + i).permutation(err)
        null_ratio.append(np.median(held_out_ratios(bmu, shuffled, num_neurons, k, cfg)))
        p_, e_ = overlap_correlations(bmu, shuffled, purity, entropy, occ, support, num_neurons)
        null_p.append(p_); null_e.append(e_)
    null_ratio, null_p, null_e = map(np.asarray, (null_ratio, null_p, null_e))

    return {
        "hotspot_concentration_held_out": {
            "n_hotspots": k,
            "n_ratios": int(ratios.size),
            "median_ratio": obs_ratio,
            "ratio_p2_5": float(np.percentile(ratios, 2.5)),
            "ratio_p97_5": float(np.percentile(ratios, 97.5)),
            "chance_median_ratio": float(np.median(null_ratio)),
            "chance_p95_ratio": float(np.percentile(null_ratio, 95)),
            "p_value_vs_chance": p_upper(obs_ratio, null_ratio),
        },
        "overlap_regions": {
            "occupied_neurons": int(occ.sum()),
            "spearman_error_rate_vs_purity": obs_p,
            "p_value_purity_negative": p_lower(obs_p, null_p),
            "spearman_error_rate_vs_entropy": obs_e,
            "p_value_entropy_positive": p_upper(obs_e, null_e),
            "chance_purity_rho_p5": float(np.nanpercentile(null_p, 5)),
            "chance_entropy_rho_p95": float(np.nanpercentile(null_e, 95)),
        },
        "settings": {"n_splits": cfg.n_splits, "n_permutations": cfg.n_permutations,
                     "hotspot_fraction": cfg.hotspot_fraction, "min_hotspots": cfg.min_hotspots,
                     "split_seed_base": cfg.split_seed_base, "perm_seed_base": cfg.perm_seed_base},
    }


def run(config_path: str | None) -> dict:
    settings = load_analysis_settings(config_path, REPO_ROOT)
    model_hash = verify_selected_model(settings)
    cfg = robustness_settings(analysis_config(settings.config_path, REPO_ROOT))
    out_path = robustness_dir(settings.config_path, REPO_ROOT) / f"{settings.selected_model}_error_pattern_robustness.json"
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists. Results are never overwritten; "
            "move or delete that file to redo the robustness statistics."
        )

    som = load_selected_som(settings.model_path, settings.grid_height, settings.grid_width, settings.feature_dim)
    val = load_split("val", embeddings_dir=settings.embeddings_dir, feature_dim=settings.feature_dim)
    clusters, cluster_distances, _, _ = som.cluster_data(val["features"])
    bmu, distance = reconstruct_assignments(clusters, cluster_distances, len(val["features"]))
    err = ~val["correct"].astype(bool)

    result = {
        "selected_model": settings.selected_model,
        "model_sha256": model_hash,
        "config": settings.config_path.name,
        "som_grid": [settings.grid_height, settings.grid_width],
        "split": "val",
        "validation_samples": int(err.size),
        "validation_errors": int(err.sum()),
        "test_evaluated": False,
        "testable": bool(err.sum() >= cfg.min_errors),
        "secondary_distance": {
            "mean_bmu_distance_error": float(distance[err].mean()),
            "mean_bmu_distance_correct": float(distance[~err].mean()),
            "median_bmu_distance_error": float(np.median(distance[err])),
            "median_bmu_distance_correct": float(np.median(distance[~err])),
        },
    }
    if result["testable"]:
        result.update(analyse(bmu, err, val["labels"], settings.num_neurons, settings.num_classes, cfg))
    else:
        logger.warning("Only %d validation errors (< %d): error-pattern claims not testable.", err.sum(), cfg.min_errors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("Error-pattern robustness statistics written to %s", out_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Error-pattern (error-geography) statistics for one SOM.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
