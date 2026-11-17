"""Stage 2 of the Optuna SOM tuning: check the chosen setting on new SOM seeds.

The search used one SOM seed, so a setting can win partly by luck of that seed. Here the
Optuna choice and the 3-point search choice (PREREGISTRATION_grid_comparison.md) are both
retrained on the same new seeds (not the search seed) and compared on validation QE, TE1
and TE1+2 as paired differences with t-intervals. Rules: OPTUNA_TUNING_PROTOCOL.md section 6.

Usage (from the code root, after the grid's search run has finished):
  python src/v1_mnist/component/p2_som_training_visualization/som_nnsom_training/optuna_confirmation.py --grid 15x15
  ... --grid 15x15 --smoke-test     # check the code after a smoke search run
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

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.optuna_common import (
    DEFAULT_CONFIG,
    PROTOCOL,
    load_frozen_inputs,
    load_tuning_config,
    new_run_dir,
    resolve,
    train_and_score,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.optuna_selection import (
    build_run_id,
    metric_verdict,
    overall_verdict,
    paired_difference,
    parse_grid,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.run_records import sha256_file
from src.v1_mnist.component.utils.logging import setup_logger

METRICS = ("val_qe", "val_te1_pct", "val_te12_pct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--grid", required=True, help="A grid whose search run has finished, e.g. 15x15.")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_tuning_config(args.config)
    opt, conf, st = cfg["optuna"], cfg["confirmation"], cfg["smoke_test"]
    smoke = args.smoke_test
    grid_name = st["grid"] if smoke else args.grid
    n_trials = st["n_trials"] if smoke else opt["n_trials"]
    search_id = build_run_id(grid_name, n_trials, opt["sampler_seed"], cfg["run"]["revision"], smoke)
    selection = json.loads((resolve(cfg["paths"]["output_dir"]) / search_id / "selection.json")
                           .read_text(encoding="utf-8"))
    seeds = st["confirmation_seeds"] if smoke else conf["seeds"]
    if opt["search_seed"] in seeds:
        raise ValueError("Confirmation seeds must not include the search seed.")
    reference = st["reference"] if smoke else cfg["reference_settings"][grid_name]
    settings = {
        "three_point": (int(reference["init_neighborhood"]), int(reference["epochs"])),
        "optuna": (int(selection["selected"]["init_neighborhood"]), int(selection["selected"]["epochs"])),
    }

    run_id = f"som_optuna_confirm_{grid_name}_seeds{len(seeds)}_{cfg['run']['revision']}" + ("_smoke" if smoke else "")
    out_dir = new_run_dir(cfg, run_id)
    logger = setup_logger("optuna_confirmation", log_file=out_dir / "run.log")
    logger.info("Run ID %s | search run %s | settings %s | seeds %s", run_id, search_id, settings, seeds)
    inputs = load_frozen_inputs(cfg, smoke)
    grid = parse_grid(grid_name)

    rows: list[dict[str, Any]] = []
    same = settings["optuna"] == settings["three_point"]
    for seed in seeds:
        for name, (nb, epochs) in settings.items():
            if same and name == "optuna":
                continue  # identical setting: train once per seed
            scores = train_and_score(inputs, grid, nb, epochs, seed)
            rows.append({"setting": name, "init_neighborhood": nb, "epochs": epochs, "seed": seed, **scores})
            logger.info("seed %d %s (nb=%d, epochs=%d): QE %.4f TE1 %.2f%% TE1+2 %.2f%%", seed, name, nb,
                        epochs, scores["val_qe"], scores["val_te1_pct"], scores["val_te12_pct"])
    with (out_dir / "confirmation_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result: dict[str, Any] = {"run_id": run_id, "search_run_id": search_id, "grid": grid_name,
                              "settings": {k: {"init_neighborhood": v[0], "epochs": v[1]} for k, v in settings.items()},
                              "seeds": seeds, "ci_level": conf["ci_level"]}
    if same:
        result["overall"] = "same setting: Optuna chose the 3-point search setting"
    else:
        by = {(r["setting"], r["seed"]): r for r in rows}
        verdicts, per_metric = [], {}
        for m in METRICS:
            diff = paired_difference([by[("optuna", s)][m] - by[("three_point", s)][m] for s in seeds],
                                     conf["ci_level"])
            diff["verdict"] = metric_verdict(diff["ci_low"], diff["ci_high"])
            verdicts.append(diff["verdict"])
            per_metric[m] = diff
        result["optuna_minus_three_point"] = per_metric
        result["overall"] = overall_verdict(verdicts)
    result.update({"protocol_sha256": sha256_file(PROTOCOL), **inputs.hashes,
                   "labels_used": False, "test_evaluated": False, "smoke_test": smoke})
    (out_dir / "confirmation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Overall: %s. Written to %s", result["overall"], out_dir)


if __name__ == "__main__":
    main()
