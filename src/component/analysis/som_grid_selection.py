"""Reproduce the frozen Week 3 SOM grid-selection decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

CANDIDATES = {
    "10x10": (
        REPO_ROOT
        / "outputs/week_3/logs/"
        "som_10x10_seed42_gridstudy_metrics.json"
    ),
    "15x15": (
        REPO_ROOT
        / "outputs/week_3/logs/"
        "som_15x15_seed42_attempt2_metrics.json"
    ),
    "20x20": (
        REPO_ROOT
        / "outputs/week_3/logs/"
        "som_20x20_seed42_gridstudy_metrics.json"
    ),
}

OUTPUT_DIR = REPO_ROOT / "outputs/week_3/tables"
RESULT_DOC = (
    REPO_ROOT
    / "docs/week_3/SOM_GRID_SELECTION_RESULT.md"
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

    if winner["grid"] != "20x20":
        raise RuntimeError(
            "Unexpected selection result. Review metrics and "
            "the frozen selection protocol before proceeding."
        )

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
            "docs/week_3/"
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
        "selected_grid": winner["grid"],
        "selected_rank_sum": winner["rank_sum"],
        "selected_model": (
            "outputs/week_3/som_models/"
            "som_20x20_seed42_gridstudy"
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

    doc = f"""# Week 3 SOM Grid-Selection Result

## Decision

**Selected development SOM: 20x20, seed 42.**

Selection followed the grid-selection protocol frozen before the
10x10 and 20x20 results were observed.

The final test split was not evaluated or used in this decision.

## Quantitative comparison

| Grid | Validation QE | QE rank | Validation TE1 (%) | TE1 rank | Validation TE1+2 (%) | TE1+2 rank | Validation occupancy | Rank sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10x10 | {by_grid["10x10"]["validation_qe"]:.6f} | {by_grid["10x10"]["qe_rank"]} | {by_grid["10x10"]["validation_te1_percent"]:.4f} | {by_grid["10x10"]["te1_rank"]} | {by_grid["10x10"]["validation_te1_plus_2_percent"]:.4f} | {by_grid["10x10"]["te1_plus_2_rank"]} | {by_grid["10x10"]["validation_occupancy"]:.4f} | {by_grid["10x10"]["rank_sum"]} |
| 15x15 | {by_grid["15x15"]["validation_qe"]:.6f} | {by_grid["15x15"]["qe_rank"]} | {by_grid["15x15"]["validation_te1_percent"]:.4f} | {by_grid["15x15"]["te1_rank"]} | {by_grid["15x15"]["validation_te1_plus_2_percent"]:.4f} | {by_grid["15x15"]["te1_plus_2_rank"]} | {by_grid["15x15"]["validation_occupancy"]:.4f} | {by_grid["15x15"]["rank_sum"]} |
| 20x20 | {by_grid["20x20"]["validation_qe"]:.6f} | {by_grid["20x20"]["qe_rank"]} | {by_grid["20x20"]["validation_te1_percent"]:.4f} | {by_grid["20x20"]["te1_rank"]} | {by_grid["20x20"]["validation_te1_plus_2_percent"]:.4f} | {by_grid["20x20"]["te1_plus_2_rank"]} | {by_grid["20x20"]["validation_occupancy"]:.4f} | **{by_grid["20x20"]["rank_sum"]}** |

## Interpretation

The metrics disagree rather than uniformly favoring one grid.

The 20x20 candidate has the lowest validation quantization error and
the lowest first-order topological error.

The 10x10 candidate has the lowest first+second-order topological
error and the highest validation occupancy.

Following the prospectively frozen equal-weight rank-sum rule, 20x20
has the lowest total rank and is therefore selected.

The occupancy reduction at 20x20 is retained as a structural
trade-off to report in subsequent analysis rather than being hidden
or used post hoc to change the selection rule.

## Selected artifact

`outputs/week_3/som_models/som_20x20_seed42_gridstudy`

## Next stage

Use the selected SOM for BMU assignment, RQ1 representation analysis,
and RQ2 error-geography analysis.

OOD/novelty evaluation remains out of scope until the Week 3
representation and error-geography analyses are frozen.
"""

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
    print(f'SELECTED GRID: {winner["grid"]}')
    print("TEST USED FOR SELECTION: NO")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"DOC:  {RESULT_DOC}")


if __name__ == "__main__":
    main()
