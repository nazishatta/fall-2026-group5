"""Stage 1 (equal tuning budget, labels not used) and Stage 2 (bootstrap maps), sections 4 and 5.

Each map is written as ``models/<map_id>.npz`` (weights and scaler, kept with DVC) plus a small
``records/<map_id>.json`` (settings, fingerprints, validation measures). A finished map is skipped, so a
stage can be re-run after an interruption; a model without its record stops the run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.endpoint import (
    map_metrics,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.fast_trainer import (
    train_som_fast,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.settings import (
    GridSettings,
    boot_id,
    tune_id,
)
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.som_io import sha256_file

log = logging.getLogger("v1_mnist.grid_comparison")
CHOICE_MEASURES = ("qe_nnsom", "te1", "te12")   # validation features only, never labels (section 4)


def _done(settings: GridSettings, map_id: str) -> bool:
    model, record = settings.models_dir / f"{map_id}.npz", settings.records_dir / f"{map_id}.json"
    if record.is_file():
        saved = json.loads(record.read_text(encoding="utf-8"))
        if not model.is_file() or sha256_file(model) != saved["model_sha256"]:
            raise RuntimeError(f"{map_id}: the model is missing or differs from its record.")
        return True
    if model.is_file():
        raise RuntimeError(f"{model} exists without a record: set a new run.revision.")
    return False


def train_map(settings: GridSettings, data: dict[str, np.ndarray], stage: str, grid: int, nb: int,
              b: int | None) -> None:
    """Train, save and record one map."""
    if stage == "tune":
        map_id, idx = tune_id(grid, nb, settings.epochs, settings.tune_seed), np.arange(len(data["x_train"]))
        seed, mode = settings.tune_seed, "exact"
    else:
        map_id, seed, mode = boot_id(grid, nb, settings.epochs, int(b)), int(b), "fast"
        n = len(data["x_train"])
        idx = np.random.default_rng(settings.resample_seed_base + int(b)).integers(0, n, n)
    if _done(settings, map_id):
        log.info("%s: already done, skipped.", map_id)
        return
    x = data["x_train"][idx]
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(x)     # refitted on the map's own training rows
    start = time.time()
    som, _ = train_som_fast(x, grid, grid, nb, settings.epochs, settings.epochs, scaler.transform, seed,
                            rng_mode=mode, record_history=False)
    seconds = time.time() - start
    weights = np.asarray(som.w, dtype=np.float64)
    model = settings.models_dir / f"{map_id}.npz"
    model.parent.mkdir(parents=True, exist_ok=True)
    with model.open("xb") as handle:
        np.savez(handle, W=weights, scaler_min=scaler.min_, scaler_scale=scaler.scale_, grid=grid, nb=nb)
    measures = map_metrics(weights, grid, scaler.transform(data["x_val"]), data["y_val"], settings.num_classes)
    write_json(settings.records_dir / f"{map_id}.json", {
        "map_id": map_id, "stage": stage, "grid": grid, "init_neighborhood": nb, "epochs": settings.epochs,
        "seed": seed, "rng_mode": mode, "bootstrap": b,
        "resample_sha256": hashlib.sha256(idx.tobytes()).hexdigest()[:16],
        "model_file": f"models/{map_id}.npz", "model_sha256": sha256_file(model),
        "train_seconds": round(seconds, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **measures,
    })
    log.info("%s: trained in %.0f s (validation QE %.4f, TE1 %.2f%%)", map_id, seconds,
             measures["qe_nnsom"], measures["te1"])


def records(settings: GridSettings, stage: str) -> list[dict[str, Any]]:
    """The records of one stage's maps."""
    found = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(settings.records_dir.glob(f"{stage}_*.json"))]
    return [r for r in found if r["stage"] == stage]


def run_tune(settings: GridSettings, data: dict[str, np.ndarray]) -> None:
    for grid in settings.grids:
        for nb in settings.candidates(grid):
            train_map(settings, data, "tune", grid, nb, None)


def run_choose(settings: GridSettings) -> dict[int, int]:
    """Within each grid: equal-weight rank sum of validation QE, TE1 and TE1+2; ties go to the smaller nb."""
    rows = records(settings, "tune")
    choice: dict[int, int] = {}
    table = []
    for grid in settings.grids:
        rs = sorted((r for r in rows if r["grid"] == grid), key=lambda r: r["init_neighborhood"])
        if [r["init_neighborhood"] for r in rs] != sorted(settings.candidates(grid)):
            raise RuntimeError(f"{grid}x{grid}: Stage 1 maps missing; run the 'tune' stage first.")
        ranks = {m: rankdata([r[m] for r in rs], method="min") for m in CHOICE_MEASURES}
        for i, r in enumerate(rs):
            r["rank_sum"] = int(sum(ranks[m][i] for m in CHOICE_MEASURES))
            table.append({k: r[k] for k in ("grid", "init_neighborhood", *CHOICE_MEASURES, "occupancy", "rank_sum")})
        best = min(rs, key=lambda r: (r["rank_sum"], r["init_neighborhood"]))
        choice[grid] = int(best["init_neighborhood"])
        log.info("%dx%d: chosen neighbourhood %d", grid, grid, choice[grid])
    if settings.choice_path.is_file():
        saved = {int(k): v for k, v in json.loads(settings.choice_path.read_text(encoding="utf-8")).items()}
        if saved != choice:
            raise RuntimeError(f"Stage 1 choice {choice} differs from the saved {saved}.")
        return choice
    write_json(settings.choice_path, {str(k): v for k, v in choice.items()})
    settings.tables_dir.mkdir(parents=True, exist_ok=True)
    with (settings.tables_dir / "stage1.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    return choice


def load_choice(settings: GridSettings) -> dict[int, int]:
    if not settings.choice_path.is_file():
        raise FileNotFoundError(f"{settings.choice_path} is missing: run the 'choose' stage first.")
    return {int(k): int(v) for k, v in json.loads(settings.choice_path.read_text(encoding="utf-8")).items()}


def run_boot(settings: GridSettings, data: dict[str, np.ndarray]) -> None:
    """The same resample b for every grid (paired design); trainer seed b, fast random stream."""
    choice = load_choice(settings)
    for b in range(settings.bootstrap_maps):
        for grid in settings.grids:
            train_map(settings, data, "boot", grid, choice[grid], b)
