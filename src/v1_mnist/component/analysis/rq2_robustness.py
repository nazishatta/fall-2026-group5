"""RQ2 error-geography statistics for one SOM, judged against "errors placed at random".

Rules: docs/som/RQ2_ROBUSTNESS_CRITERIA.md. Nothing here is specific to MNIST,
the grid size or the CNN: every claim is compared with the same chance baseline,
obtained by shuffling which validation images are errors (keeping each image's
neuron and class), so the same code and rules apply to any dataset.

Claim A, error hotspots (held-out):
  - hotspots = top k occupied neurons by 95% Wilson lower bound of error rate,
    k = max(5, round(5% of neurons)); chosen on one half of the validation set.
  - ratio = hotspots' share of the other half's errors / their share of its images.
  - statistic = median ratio over N_SPLITS stratified splits x 2 directions.
  - chance baseline: the same statistic after shuffling error labels (N_PERM times).
Claim B, overlap regions:
  - Spearman of neuron error rate with class purity (expect negative) and with
    normalized class entropy (expect positive), over occupied neurons,
    each compared with the same shuffled-label baseline.
Secondary (novelty signal, reported only): BMU distance, errors vs correct.

Usage (from the code root):
  python src/v1_mnist/component/analysis/rq2_robustness.py [--config CONFIG] [--overwrite]
Writes outputs/v1_mnist/som/analysis/<selected_model>_rq2_robustness.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.analysis.io import load_selected_som, load_split, reconstruct_assignments
from src.v1_mnist.component.analysis.metrics import CLASS_COUNT, normalized_entropy
from src.v1_mnist.component.analysis.run_settings import DEFAULT_CONFIG, load_rq_settings, verify_selected_model
from src.v1_mnist.component.utils.logging import get_logger

logger = get_logger("v1_mnist.analysis.rq2_robustness")

HOTSPOT_FRACTION = 0.05
MIN_HOTSPOTS = 5
N_SPLITS = 100
N_PERM = 200
MIN_ERRORS = 20
SPLIT_SEED_BASE = 7000
PERM_SEED_BASE = 9000
Z95 = 1.959963984540054


def n_hotspots(num_neurons: int) -> int:
    return max(MIN_HOTSPOTS, int(round(HOTSPOT_FRACTION * num_neurons)))


def wilson_lower(errors: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Vectorized 95% Wilson lower bound (same formula as metrics.wilson_lower_bound)."""
    n = np.maximum(total, 1).astype(float)
    p = errors / n
    z2 = Z95 * Z95
    den = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / den
    half = Z95 * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return np.where(total > 0, np.maximum(0.0, center - half), -np.inf)


def top_hotspots(bmu: np.ndarray, err: np.ndarray, num_neurons: int, k: int) -> np.ndarray:
    """Top-k occupied neurons: Wilson bound desc, then errors desc, support desc, index asc."""
    support = np.bincount(bmu, minlength=num_neurons)
    errors = np.bincount(bmu, weights=err, minlength=num_neurons)
    lb = wilson_lower(errors, support)
    order = np.lexsort((np.arange(num_neurons), -support, -errors, -lb))
    order = order[support[order] > 0]
    return order[:k]


def stratified_halves(err: np.ndarray, rep: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SPLIT_SEED_BASE + rep)
    a, b = [], []
    for value in (True, False):
        idx = rng.permutation(np.flatnonzero(err == value))
        h = len(idx) // 2
        a.append(idx[:h]); b.append(idx[h:])
    return np.concatenate(a), np.concatenate(b)


def held_out_ratios(bmu: np.ndarray, err: np.ndarray, num_neurons: int, k: int) -> np.ndarray:
    ratios = []
    for rep in range(N_SPLITS):
        a, b = stratified_halves(err, rep)
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


def analyse(bmu: np.ndarray, err_bool: np.ndarray, labels: np.ndarray, num_neurons: int) -> dict:
    err = err_bool.astype(float)
    k = n_hotspots(num_neurons)
    support = np.bincount(bmu, minlength=num_neurons)
    occ = support > 0
    counts = np.zeros((num_neurons, CLASS_COUNT))
    np.add.at(counts, (bmu, labels.astype(int)), 1)
    purity = counts[occ].max(1) / support[occ]
    entropy = np.array([normalized_entropy(c) for c in counts[occ]])

    ratios = held_out_ratios(bmu, err, num_neurons, k)
    obs_a = float(np.median(ratios))
    obs_p, obs_e = overlap_correlations(bmu, err, purity, entropy, occ, support, num_neurons)

    null_a, null_p, null_e = [], [], []
    for i in range(N_PERM):
        shuffled = np.random.default_rng(PERM_SEED_BASE + i).permutation(err)
        null_a.append(np.median(held_out_ratios(bmu, shuffled, num_neurons, k)))
        p_, e_ = overlap_correlations(bmu, shuffled, purity, entropy, occ, support, num_neurons)
        null_p.append(p_); null_e.append(e_)
    null_a, null_p, null_e = map(np.asarray, (null_a, null_p, null_e))

    return {
        "claim_a_hotspots_held_out": {
            "n_hotspots": k,
            "n_ratios": int(ratios.size),
            "median_ratio": obs_a,
            "ratio_p2_5": float(np.percentile(ratios, 2.5)),
            "ratio_p97_5": float(np.percentile(ratios, 97.5)),
            "chance_median_ratio": float(np.median(null_a)),
            "chance_p95_ratio": float(np.percentile(null_a, 95)),
            "p_value_vs_chance": p_upper(obs_a, null_a),
        },
        "claim_b_overlap_regions": {
            "occupied_neurons": int(occ.sum()),
            "spearman_error_rate_vs_purity": obs_p,
            "p_value_purity_negative": p_lower(obs_p, null_p),
            "spearman_error_rate_vs_entropy": obs_e,
            "p_value_entropy_positive": p_upper(obs_e, null_e),
            "chance_purity_rho_p5": float(np.nanpercentile(null_p, 5)),
            "chance_entropy_rho_p95": float(np.nanpercentile(null_e, 95)),
        },
        "settings": {"n_splits": N_SPLITS, "n_permutations": N_PERM,
                     "hotspot_fraction": HOTSPOT_FRACTION, "min_hotspots": MIN_HOTSPOTS,
                     "split_seed_base": SPLIT_SEED_BASE, "perm_seed_base": PERM_SEED_BASE},
    }


def run(config_path: str | None, overwrite: bool = False) -> dict:
    settings = load_rq_settings(config_path, REPO_ROOT)
    model_hash = verify_selected_model(settings)
    out_path = REPO_ROOT / "outputs/v1_mnist/som/analysis" / f"{settings.selected_model}_rq2_robustness.json"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it.")

    som = load_selected_som(settings.model_path, settings.grid_height, settings.grid_width)
    val = load_split("val", embeddings_dir=settings.embeddings_dir)
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
        "testable": bool(err.sum() >= MIN_ERRORS),
        "secondary_distance": {
            "mean_bmu_distance_error": float(distance[err].mean()),
            "mean_bmu_distance_correct": float(distance[~err].mean()),
            "median_bmu_distance_error": float(np.median(distance[err])),
            "median_bmu_distance_correct": float(np.median(distance[~err])),
        },
    }
    if result["testable"]:
        result.update(analyse(bmu, err, val["labels"], settings.num_neurons))
    else:
        logger.warning("Only %d validation errors (< %d): RQ2 claims not testable.", err.sum(), MIN_ERRORS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("RQ2 robustness statistics written to %s", out_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 error-geography statistics for one SOM.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.overwrite)


if __name__ == "__main__":
    main()
