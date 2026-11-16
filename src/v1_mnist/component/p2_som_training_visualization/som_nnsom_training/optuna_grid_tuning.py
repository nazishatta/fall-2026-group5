"""Stage 1 of the Optuna SOM tuning: the search, one grid per run (validation features only).

Optuna's TPE sampler (https://optuna.org) tunes two settings of the project's batch SOM
trainer (NNSOM's SOM has no learning rate):
  - the starting neighbourhood, searched as a fraction of the grid's side length, so every
    grid has the same search space (as in PREREGISTRATION_grid_comparison.md);
  - the number of epochs (steps = epochs).
Each trial is scored by validation QE and TE1 (two objectives, both minimised). A setting
that was already trained is reused, not retrained (same seed, same result). The setting
for the grid is chosen by optuna_selection.select_trial. Rules: OPTUNA_TUNING_PROTOCOL.md.

Usage (from the code root):
  python src/v1_mnist/component/p2_som_training_visualization/som_nnsom_training/optuna_grid_tuning.py --grid 15x15
  ... --grid 15x15 --smoke-test     # 5x5 map, small subsets, 3 short trials, separate run ID
  ... --summarize                   # one table from the three finished grids
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optuna

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.optuna_common import (
    DEFAULT_CONFIG,
    PROTOCOL,
    FrozenInputs,
    load_frozen_inputs,
    load_tuning_config,
    neighbourhood_from_fraction,
    new_run_dir,
    resolve,
    train_and_score,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.optuna_selection import (
    TrialResult,
    build_run_id,
    parse_grid,
    select_trial,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.run_records import sha256_file
from src.v1_mnist.component.utils.logging import setup_logger

TRIAL_FIELDS = ["number", "nb_fraction", "init_neighborhood", "epochs", "val_qe", "val_te1_pct",
                "val_te12_pct", "val_occupancy", "train_seconds", "reused"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--grid", help="One of the grids in the config, e.g. 15x15.")
    group.add_argument("--summarize", action="store_true", help="Combine the finished grids into one table.")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny run to check the code end to end.")
    return parser.parse_args()


def make_objective(inputs: FrozenInputs, grid: tuple[int, int], space: dict[str, Any],
                   seed: int, logger: Any) -> Any:
    """Optuna objective: (validation QE, validation TE1 %) for one suggested setting."""
    side = max(grid)
    done: dict[tuple[int, int], dict[str, float]] = {}

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        fraction = trial.suggest_float("nb_fraction", space["nb_fraction_min"], space["nb_fraction_max"])
        epochs = trial.suggest_int("epochs", space["epochs_min"], space["epochs_max"], step=space["epochs_step"])
        nb = neighbourhood_from_fraction(fraction, side)
        key = (nb, epochs)
        reused = key in done
        if not reused:
            done[key] = train_and_score(inputs, grid, nb, epochs, seed)
        scores = done[key]
        trial.set_user_attr("init_neighborhood", nb)
        trial.set_user_attr("reused", reused)
        for name in ("val_te12_pct", "val_occupancy", "train_seconds"):
            trial.set_user_attr(name, scores[name])
        logger.info("Trial %d: fraction %.3f -> nb=%d, epochs=%d | val QE %.4f TE1 %.2f%% TE1+2 %.2f%%%s",
                    trial.number, fraction, nb, epochs, scores["val_qe"], scores["val_te1_pct"],
                    scores["val_te12_pct"], " (reused)" if reused else f" | {scores['train_seconds']:.0f}s")
        return scores["val_qe"], scores["val_te1_pct"]

    return objective


def trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    """One row per finished trial."""
    return [{
        "number": t.number, "nb_fraction": t.params["nb_fraction"],
        "init_neighborhood": t.user_attrs["init_neighborhood"], "epochs": t.params["epochs"],
        "val_qe": t.values[0], "val_te1_pct": t.values[1], "val_te12_pct": t.user_attrs["val_te12_pct"],
        "val_occupancy": t.user_attrs["val_occupancy"], "train_seconds": t.user_attrs["train_seconds"],
        "reused": t.user_attrs["reused"],
    } for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def run_grid(cfg: dict[str, Any], grid_name: str, smoke: bool) -> None:
    opt, st = cfg["optuna"], cfg["smoke_test"]
    if not smoke and grid_name not in opt["grids"]:
        raise ValueError(f"Grid {grid_name} is not in the config ({opt['grids']}).")
    grid_name = st["grid"] if smoke else grid_name
    n_trials = st["n_trials"] if smoke else opt["n_trials"]
    run_id = build_run_id(grid_name, n_trials, opt["sampler_seed"], cfg["run"]["revision"], smoke)
    out_dir = new_run_dir(cfg, run_id)
    logger = setup_logger("optuna_grid_tuning", log_file=out_dir / "run.log")
    logger.info("Run ID %s | grid %s | %d trials | smoke=%s", run_id, grid_name, n_trials, smoke)

    inputs = load_frozen_inputs(cfg, smoke)
    space = dict(opt["search_space"])
    if smoke:
        space.update({k: st[k] for k in ("epochs_min", "epochs_max", "epochs_step")})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=opt["sampler_seed"], n_startup_trials=opt["n_startup_trials"])
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler, study_name=run_id)
    study.optimize(make_objective(inputs, parse_grid(grid_name), space, opt["search_seed"], logger),
                   n_trials=n_trials)

    rows = trial_rows(study)
    with (out_dir / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    chosen = select_trial([TrialResult(**{k: r[k] for k in TrialResult.__dataclass_fields__}) for r in rows])
    result = {
        "run_id": run_id, "grid": grid_name, "n_trials": n_trials, "smoke_test": smoke,
        "distinct_settings_trained": sum(not r["reused"] for r in rows),
        "selected": {"trial": chosen.number, "init_neighborhood": chosen.init_neighborhood,
                     "epochs": chosen.epochs, "val_qe": chosen.val_qe, "val_te1_pct": chosen.val_te1_pct,
                     "val_te12_pct": chosen.val_te12_pct},
        "pareto_trials": [t.number for t in study.best_trials],
        "search_space": space, "sampler": "TPESampler", "sampler_seed": opt["sampler_seed"],
        "n_startup_trials": opt["n_startup_trials"], "search_seed": opt["search_seed"],
        "optuna_version": optuna.__version__, "protocol_sha256": sha256_file(PROTOCOL), **inputs.hashes,
        "labels_used": False, "test_evaluated": False,
    }
    (out_dir / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Selected trial %d: nb=%d epochs=%d (%d distinct settings trained). Written to %s",
                chosen.number, chosen.init_neighborhood, chosen.epochs, result["distinct_settings_trained"], out_dir)


def summarize(cfg: dict[str, Any]) -> None:
    """One row per grid from the finished (non-smoke) search runs."""
    logger = setup_logger("optuna_grid_tuning")
    opt, rows = cfg["optuna"], []
    for grid in opt["grids"]:
        run_id = build_run_id(grid, opt["n_trials"], opt["sampler_seed"], cfg["run"]["revision"])
        sel = json.loads((resolve(cfg["paths"]["output_dir"]) / run_id / "selection.json").read_text(encoding="utf-8"))
        ref = cfg["reference_settings"][grid]
        rows.append({"grid": grid, "run_id": run_id, "reference_nb": ref["init_neighborhood"],
                     "reference_epochs": ref["epochs"], **sel["selected"]})
    path = resolve(cfg["paths"]["summary_table"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Summary written to %s", path)


def main() -> None:
    args = parse_args()
    cfg = load_tuning_config(args.config)
    if args.summarize:
        summarize(cfg)
    else:
        run_grid(cfg, args.grid, args.smoke_test)


if __name__ == "__main__":
    main()
