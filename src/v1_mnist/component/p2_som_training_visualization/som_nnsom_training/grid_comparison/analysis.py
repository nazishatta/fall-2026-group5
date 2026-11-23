"""The primary endpoint and the secondary and exploratory results (sections 6 and 7), and the
reproduction checks against the original MNIST run.

Ported unchanged in its maths from the original ``analyze.py``. Added after the MNIST results were
known (labelled exploratory): the 20x20 minus 10x10 comparisons.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.endpoint import (
    bmus,
    crossfit,
    map_metrics,
    split_indices,
    summarise,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.provenance import (
    low_confidence,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.settings import (
    METRICS,
    GridSettings,
    boot_id,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.stages import (
    records,
)
from src.v1_mnist.component.utils.run_provenance import ensure_not_done, write_json

log = logging.getLogger("v1_mnist.grid_comparison")
DESCRIPTIVE = ("qe_nnsom", "qe_per_image", "te1", "te12", "occupancy", "purity", "neighbour_disagreement")
MapPath = Callable[[int, int, int], Path]   # (grid, neighbourhood, bootstrap map) -> .npz file


def scaled_val(path: Path, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A saved map's weights and the validation features through its own scaler."""
    z = np.load(path)
    return z["W"], x_val * z["scaler_scale"] + z["scaler_min"]


def endpoint_rows(settings: GridSettings, data: dict[str, np.ndarray], lowconf: np.ndarray,
                  choice: dict[int, int], map_path: MapPath) -> list[dict[str, float]]:
    """One row per repeat: every grid's held-out AUROC and average precision for both targets."""
    cells = {}
    for grid in settings.grids:
        for b in range(settings.bootstrap_maps):
            weights, xs = scaled_val(map_path(grid, choice[grid], b), data["x_val"])
            cells[grid, b] = bmus(weights, xs)[0]
    rows = []
    for rep in range(settings.reps):
        half_a, half_b = split_indices(data["err"], rep, settings.split_seed_base)
        b = rep % settings.bootstrap_maps
        row: dict[str, float] = {"rep": rep, "bootstrap_map": b}
        for grid in settings.grids:
            n = grid * grid
            row[f"auroc_err_{grid}"], row[f"ap_err_{grid}"] = crossfit(cells[grid, b], data["err"], half_a, half_b, n)
            row[f"auroc_lowconf_{grid}"], row[f"ap_lowconf_{grid}"] = crossfit(
                cells[grid, b], lowconf, half_a, half_b, n)
        rows.append(row)
    return rows


def summary(settings: GridSettings, rows: list[dict[str, float]],
            descriptive: dict[int, dict[str, dict[str, float]]], choice: dict[int, int]) -> dict[str, Any]:
    per_grid = {}
    for grid in settings.grids:
        per_grid[grid] = {}
        for metric in METRICS:
            v = np.array([r[f"{metric}_{grid}"] for r in rows])
            per_grid[grid][metric] = {"mean": float(v.mean()), "p2_5": float(np.percentile(v, 2.5)),
                                      "p97_5": float(np.percentile(v, 97.5))}
    return {
        "primary": summarise(rows, settings.primary, settings.margin),
        "secondary": [summarise(rows, c, settings.margin) for c in settings.secondary],
        "exploratory": [summarise(rows, c, settings.margin) for c in settings.exploratory],
        "per_grid": per_grid,
        "descriptive": descriptive,
        "chosen_neighbourhood": choice,
        "reps": settings.reps,
        "bootstrap_maps": settings.bootstrap_maps,
        "margin": settings.margin,
    }


def descriptive_from(measures: dict[int, list[dict[str, float]]]) -> dict[int, dict[str, dict[str, float]]]:
    """Mean and SD of each map measure over a grid's bootstrap maps."""
    return {g: {m: {"mean": float(np.mean([x[m] for x in rs])), "sd": float(np.std([x[m] for x in rs], ddof=1))}
                for m in DESCRIPTIVE} for g, rs in measures.items()}


def run_analyze(settings: GridSettings, data: dict[str, np.ndarray], choice: dict[int, int]) -> dict[str, Any]:
    """Endpoint on this run's bootstrap maps; writes the repeats table and results.json."""
    ensure_not_done(settings.results_path)
    lowconf, threshold = low_confidence(data["conf"], settings.low_confidence_quantile,
                                        settings.expected["low_confidence_images"])
    boot = records(settings, "boot")
    measures = {g: [r for r in boot if r["grid"] == g] for g in settings.grids}
    for g, rs in measures.items():
        if len(rs) != settings.bootstrap_maps:
            raise RuntimeError(f"{g}x{g}: {len(rs)} bootstrap maps, expected {settings.bootstrap_maps}.")

    def path(grid: int, nb: int, b: int) -> Path:
        return settings.models_dir / f"{boot_id(grid, nb, settings.epochs, b)}.npz"

    rows = endpoint_rows(settings, data, lowconf, choice, path)
    result = summary(settings, rows, descriptive_from(measures), choice)
    result.update({"n_val": int(len(data["err"])), "n_errors": int(data["err"].sum()),
                   "n_lowconf": int(lowconf.sum()), "low_confidence_threshold": threshold,
                   "test_split_read": False, "exploratory_note": (
                       "The 20x20 minus 10x10 comparisons were added on 2026-11-21, after the MNIST results "
                       "were known; they are not part of the pre-registered decision.")})
    settings.tables_dir.mkdir(parents=True, exist_ok=True)
    with (settings.tables_dir / "endpoint_reps.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(settings.results_path, result)
    p = result["primary"]
    log.info("Primary (%s, %s): %+.4f [%+.4f, %+.4f] -> %s", p["contrast"], p["metric"], p["mean_delta"],
             p["ci_low"], p["ci_high"], p["verdict"])
    for s in result["secondary"] + result["exploratory"]:
        log.info("%s %s: %+.4f [%+.4f, %+.4f] -> %s", s["contrast"], s["metric"], s["mean_delta"],
                 s["ci_low"], s["ci_high"], s["verdict"])
    return result


def _differences(new: Any, old: Any, prefix: str = "") -> dict[str, float]:
    """Absolute difference of every number in ``old``; ``new`` must hold all of them."""
    out: dict[str, float] = {}
    if isinstance(old, dict):
        for k, v in old.items():
            if k not in new:
                raise KeyError(f"{prefix}{k} is missing from the ported results.")
            out.update(_differences(new[k], v, f"{prefix}{k}."))
    elif isinstance(old, list):
        if len(new) != len(old):
            raise ValueError(f"{prefix}: {len(new)} entries against {len(old)} in the original.")
        for i, (a, b) in enumerate(zip(new, old)):
            out.update(_differences(a, b, f"{prefix}{i}."))
    elif isinstance(old, (int, float)) and not isinstance(old, bool):
        out[prefix.rstrip(".")] = abs(float(new) - float(old))
    return out


def verify_saved_maps(settings: GridSettings, data: dict[str, np.ndarray]) -> dict[str, Any]:
    """The ported endpoint on the original run's 30 saved bootstrap maps must give its numbers."""
    rep = settings.reproduction
    if rep is None:
        raise RuntimeError("This config has no reproduction section (only MNIST has an original run).")
    out = settings.output_dir / "reproduction" / "saved_maps_check.json"
    ensure_not_done(out)
    original = json.loads(rep.results.read_text(encoding="utf-8"))
    choice = {int(k): int(v) for k, v in original["chosen_neighbourhood"].items()}
    lowconf, threshold = low_confidence(data["conf"], settings.low_confidence_quantile,
                                        settings.expected["low_confidence_images"])

    def path(grid: int, nb: int, b: int) -> Path:
        return rep.map_path(grid, nb, settings.epochs, b)

    rows = endpoint_rows(settings, data, lowconf, choice, path)
    measures: dict[int, list[dict[str, float]]] = {}
    for g in settings.grids:
        measures[g] = []
        for b in range(settings.bootstrap_maps):
            weights, xs = scaled_val(path(g, choice[g], b), data["x_val"])
            measures[g].append(map_metrics(weights, g, xs, data["y_val"], settings.num_classes))
    # Through JSON, so the grid keys are text as in the original results.json.
    ported = json.loads(json.dumps(summary(settings, rows, descriptive_from(measures), choice)))
    compared = {k: original[k] for k in ("primary", "secondary", "per_grid", "descriptive")}
    diffs = _differences(ported, compared)
    worst = max(diffs.values())
    verdicts_same = (ported["primary"]["verdict"] == original["primary"]["verdict"]
                     and [s["verdict"] for s in ported["secondary"]] == [s["verdict"] for s in original["secondary"]])
    check = {"maps_dir": rep.maps_dir_config, "numbers_compared": len(diffs), "largest_difference": worst,
             "tolerance": rep.saved_maps_tolerance, "verdicts_same": verdicts_same,
             "low_confidence_threshold": threshold,
             "passed": bool(worst <= rep.saved_maps_tolerance and verdicts_same),
             "exploratory_on_saved_maps": ported["exploratory"]}
    write_json(out, check)
    log.info("Saved-map check: %d numbers, largest difference %.3g -> %s", len(diffs), worst,
             "PASSED" if check["passed"] else "FAILED")
    if not check["passed"]:
        raise RuntimeError(f"The port does not reproduce the original numbers: see {out}.")
    return check


def verify_retrain(settings: GridSettings, result: dict[str, Any]) -> dict[str, Any]:
    """A fresh retrain must give the original verdicts and per-grid AUROC within the fixed tolerance."""
    rep = settings.reproduction
    if rep is None:
        raise RuntimeError("This config has no reproduction section.")
    out = settings.output_dir / "reproduction" / "retrain_check.json"
    ensure_not_done(out)
    original = json.loads(rep.results.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(result))   # grid keys as text, as in the files
    auroc = {str(g): abs(result["per_grid"][str(g)]["auroc_err"]["mean"]
                         - original["per_grid"][str(g)]["auroc_err"]["mean"]) for g in settings.grids}
    verdicts = {"primary": (result["primary"]["verdict"], original["primary"]["verdict"]),
                **{f"secondary_{i}": (a["verdict"], b["verdict"])
                   for i, (a, b) in enumerate(zip(result["secondary"], original["secondary"]))}}
    check = {"auroc_err_mean_difference_per_grid": auroc, "tolerance": rep.retrain_auroc_tolerance,
             "verdicts_new_vs_original": verdicts,
             "choice_same": result["chosen_neighbourhood"] == original["chosen_neighbourhood"],
             "passed": bool(max(auroc.values()) <= rep.retrain_auroc_tolerance
                            and all(a == b for a, b in verdicts.values()))}
    write_json(out, check)
    log.info("Retrain check -> %s", "PASSED" if check["passed"] else "FAILED (see %s)" % out)
    return check
