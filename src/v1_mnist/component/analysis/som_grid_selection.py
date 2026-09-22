"""Optional MNIST SOM grid-size ablation experiment.

This script compares 10x10, 15x15, and 20x20 SOM runs for research
analysis only. The main project SOM is fixed at 15x15 based on the
project's visual-interpretability design choice.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]

CANDIDATES = {
    "10x10": (
        REPO_ROOT
        / "outputs/v1_mnist/som/logs/"
        "som_10x10_seed42_gridstudy_metrics.json"
    ),
    "15x15": (
        REPO_ROOT
        / "outputs/v1_mnist/som/logs/"
        "som_15x15_seed42_attempt2_metrics.json"
    ),
    "20x20": (
        REPO_ROOT
        / "outputs/v1_mnist/som/logs/"
        "som_20x20_seed42_gridstudy_metrics.json"
    ),
}

OUTPUT_DIR = REPO_ROOT / "outputs/v1_mnist/som/tables"
RESULT_DOC = (
    REPO_ROOT
    / "outputs/v1_mnist/som/docs"
    / "SOM_GRID_SELECTION_RESULT.md"
)


def ordinal_ranks(
    rows: list[dict],
    metric: str,
) -> dict[str, int]:
    """Assign ascending ordinal ranks; lower metric value is better."""

    ordered = sorted(
        rows,
        key=lambda row: row[metric],
    )

    return {
        row["grid"]: rank
        for rank, row in enumerate(ordered, start=1)
    }


def main() -> None:
    rows = []

    for grid, path in CANDIDATES.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        if data.get("test_evaluated") is not False:
            raise RuntimeError(
                f"{grid}: test_evaluated must be false."
            )

        val = data["validation_metrics"]
        train = data["train_metrics"]

        rows.append(
            {
                "grid": grid,
                "metrics_path": str(
                    path.relative_to(REPO_ROOT)
                ),
                "validation_qe": float(
                    val["quantization_error"]
                ),
                "validation_te1_percent": float(
                    val[
                        "topological_error_1st_order_pct"
                    ]
                ),
                "validation_te1_plus_2_percent": float(
                    val[
                        "topological_error_1st_2nd_order_pct"
                    ]
                ),
                "validation_occupancy": float(
                    val["occupancy_rate"]
                ),
                "validation_occupied_neurons": int(
                    val["occupied_neurons"]
                ),
                "validation_total_neurons": int(
                    val["total_neurons"]
                ),
                "train_occupancy": float(
                    train["occupancy_rate"]
                ),
                "test_evaluated": False,
            }
        )

    metric_rank_names = {
        "validation_qe": "qe_rank",
        "validation_te1_percent": "te1_rank",
        "validation_te1_plus_2_percent": "te1_plus_2_rank",
    }

    for metric, rank_name in metric_rank_names.items():
        ranks = ordinal_ranks(rows, metric)

        for row in rows:
            row[rank_name] = ranks[row["grid"]]

    for row in rows:
        row["rank_sum"] = (
            row["qe_rank"]
            + row["te1_rank"]
            + row["te1_plus_2_rank"]
        )

    # Frozen protocol:
    # lowest rank sum wins; smaller grid only breaks a total-rank tie.
    def grid_neurons(grid: str) -> int:
        side = int(grid.split("x")[0])
        return side * side

    winner = min(
        rows,
        key=lambda row: (
            row["rank_sum"],
            grid_neurons(row["grid"]),
        ),
    )

    selected_grid = winner["grid"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "som_grid_selection.csv"
    json_path = OUTPUT_DIR / "som_grid_selection.json"

    fieldnames = list(rows[0].keys())

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "protocol": (
            "outputs/v1_mnist/som/docs/"
            "SOM_GRID_SELECTION_PROTOCOL.md"
        ),
        "selection_stage": "development",
        "candidate_seed": 42,
        "test_used_for_selection": False,
        "ranking_rule": (
            "Equal-weight ordinal rank sum over validation "
            "QE, TE1, and TE1+2; lowest total rank wins; "
            "smaller grid breaks a total-rank tie."
        ),
        "candidates": rows,
        "selected_grid": selected_grid,
        "selected_rank_sum": winner["rank_sum"],
        "selected_model": (
            f"outputs/v1_mnist/som/som_models/"
            f"som_{selected_grid.replace('x', 'x')}_seed42_gridstudy"
        ),
    }

    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    by_grid = {
        row["grid"]: row
        for row in rows
    }

    # Build the winner row highlight: bold the winning grid's rank sum
    def fmt_row(g):
        r = by_grid[g]
        rank_sum_str = (
            f'**{r["rank_sum"]}**' if g == selected_grid else str(r["rank_sum"])
        )
        return (
            f'| {g} | {r["validation_qe"]:.6f} | {r["qe_rank"]} | '
            f'{r["validation_te1_percent"]:.4f} | {r["te1_rank"]} | '
            f'{r["validation_te1_plus_2_percent"]:.4f} | {r["te1_plus_2_rank"]} | '
            f'{r["validation_occupancy"]:.4f} | {rank_sum_str} |'
        )

    doc = f"""# MNIST v1 SOM Grid-Selection Result

## Decision

**Selected development SOM: {selected_grid}, seed 42.**

Selection followed the equal-weight rank-sum protocol frozen before
any grid results were observed.

The final test split was not evaluated or used in this decision.

## Quantitative comparison

| Grid | Validation QE | QE rank | Validation TE1 (%) | TE1 rank | Validation TE1+2 (%) | TE1+2 rank | Validation occupancy | Rank sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{fmt_row("10x10")}
{fmt_row("15x15")}
{fmt_row("20x20")}

## Interpretation

The ranking rule: equal-weight ordinal rank sum over validation QE,
TE1, and TE1+2. Lowest total rank wins. Smaller grid breaks a tie.

Selected: **{selected_grid}** (rank sum = {winner["rank_sum"]})

## Selected artifact

`outputs/v1_mnist/som/som_models/som_{selected_grid}_seed42_gridstudy`

## Next stage

Use the selected SOM for BMU assignment, RQ1 representation analysis,
and RQ2 error-geography analysis.

OOD/novelty evaluation remains out of scope until the representation
and error-geography analyses are frozen.
"""

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DOC.write_text(
        doc,
        encoding="utf-8",
    )

    print("\nGRID SELECTION")
    print("=" * 60)

    for row in sorted(
        rows,
        key=lambda r: r["rank_sum"],
    ):
        print(
            f'{row["grid"]:>5} | '
            f'QE={row["validation_qe"]:.6f} '
            f'(r{row["qe_rank"]}) | '
            f'TE1={row["validation_te1_percent"]:.4f} '
            f'(r{row["te1_rank"]}) | '
            f'TE1+2={row["validation_te1_plus_2_percent"]:.4f} '
            f'(r{row["te1_plus_2_rank"]}) | '
            f'total={row["rank_sum"]}'
        )

    print()
    print(f'SELECTED GRID: {selected_grid}')
    print("TEST USED FOR SELECTION: NO")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"DOC:  {RESULT_DOC}")


if __name__ == "__main__":
    main()
