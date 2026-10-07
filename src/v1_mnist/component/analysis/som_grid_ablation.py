"""Descriptive 10x10 / 15x15 / 20x20 SOM grid ablation table.

Reads the validation metrics of the official v2 runs (or any run IDs given on
the command line) and writes one comparison table. It does NOT select a grid:
the grid decision and its reasons are recorded in
docs/som/SOM_GRID_SELECTION_PROTOCOL.md (Recorded outcome) and the
pre-registered comparison in docs/som/GRID_COMPARISON_RESULTS.md.

The original protocol's rank sum (validation QE, TE1, TE1+2) is reported for
reference only. QE falls and TE rises with grid size by construction, so the
ranks lean toward particular sizes and are not used to choose a grid.

Usage (from the code root):
  python src/v1_mnist/component/analysis/som_grid_ablation.py
  python src/v1_mnist/component/analysis/som_grid_ablation.py --run-ids A B C
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.logging import get_logger

logger = get_logger("v1_mnist.analysis.som_grid_ablation")

LOGS_DIR = REPO_ROOT / "outputs/v1_mnist/som/logs"
OUTPUT_DIR = REPO_ROOT / "outputs/v1_mnist/som/tables"
DEFAULT_RUN_IDS = [
    "som_10x10_nb7_ep250_s42_v2",
    "som_15x15_nb11_ep250_s42_v2",
    "som_20x20_nb15_ep250_s42_v2",
]
RANKED = {
    "validation_qe": "qe_rank",
    "validation_te1_percent": "te1_rank",
    "validation_te1_plus_2_percent": "te1_plus_2_rank",
}


def load_row(run_id: str) -> dict:
    """One table row from a run's metrics file."""
    path = LOGS_DIR / f"{run_id}_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("test_evaluated") is not False:
        raise RuntimeError(f"{run_id}: test_evaluated must be false.")
    som, val, train = data["som"], data["validation_metrics"], data["train_metrics"]
    return {
        "run_id": run_id,
        "grid": f'{som["grid_height"]}x{som["grid_width"]}',
        "init_neighborhood": som["init_neighborhood"],
        "epochs": som["epochs"],
        "model_sha256": (data.get("model") or {}).get("sha256", ""),
        "validation_qe": float(val["quantization_error"]),
        "validation_te1_percent": float(val["topological_error_1st_order_pct"]),
        "validation_te1_plus_2_percent": float(val["topological_error_1st_2nd_order_pct"]),
        "validation_occupancy": float(val["occupancy_rate"]),
        "train_occupancy": float(train["occupancy_rate"]),
    }


def add_reference_ranks(rows: list[dict]) -> None:
    """Original-protocol ranks (1 = lowest value), reported for reference only."""
    for metric, rank_name in RANKED.items():
        for rank, row in enumerate(sorted(rows, key=lambda r: r[metric]), start=1):
            row[rank_name] = rank
    for row in rows:
        row["reference_rank_sum"] = sum(row[r] for r in RANKED.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Descriptive SOM grid ablation table.")
    parser.add_argument("--run-ids", nargs="+", default=DEFAULT_RUN_IDS)
    args = parser.parse_args()

    rows = sorted((load_row(r) for r in args.run_ids),
                  key=lambda r: int(r["grid"].split("x")[0]))
    add_reference_ranks(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "som_grid_ablation.csv"
    json_path = OUTPUT_DIR / "som_grid_ablation.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({
        "purpose": "descriptive ablation; not used to select a grid",
        "grid_decision": "docs/som/SOM_GRID_SELECTION_PROTOCOL.md (Recorded outcome)",
        "test_evaluated": False,
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")

    for r in rows:
        logger.info("%5s nb%-2s | QE=%.6f TE1=%.4f TE1+2=%.4f occ=%.4f | reference rank sum %d",
                    r["grid"], r["init_neighborhood"], r["validation_qe"],
                    r["validation_te1_percent"], r["validation_te1_plus_2_percent"],
                    r["validation_occupancy"], r["reference_rank_sum"])
    logger.info("Descriptive only: no grid is selected here. CSV: %s", csv_path)


if __name__ == "__main__":
    main()
