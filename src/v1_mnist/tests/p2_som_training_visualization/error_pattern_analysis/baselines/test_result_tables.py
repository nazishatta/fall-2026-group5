from __future__ import annotations

import csv
from pathlib import Path

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.result_tables import (
    latex_table,
    write_report_outputs,
    write_summary_csv,
)


def _interval(mean: float) -> dict:
    return {"mean": mean, "lower": mean - 0.01, "upper": mean + 0.01}


def _results() -> dict:
    conc = {"median_ratio": 11.1, "ratio_p2_5": 6.1, "ratio_p97_5": 17.6, "p_value_vs_chance": 0.005, "n_hotspots": 11}
    return {
        "run_id": "error_pattern_baselines_demo_r1", "validation_samples": 10500, "validation_errors": 92,
        "settings": {"grid": [15, 15], "num_cells": 225, "knn_neighbors": 15, "kmeans_seeds": [0, 1]},
        "hotspot": {
            "som_cell": {"hotspot_concentration_held_out": conc,
                         "hotspot_shares": {"mean_image_share_in_hotspots": 0.016, "mean_error_share_in_hotspots": 0.18}},
            "kmeans_cell": {"median_ratio_over_seeds": 5.3, "min_ratio": 4.5, "max_ratio": 7.3, "max_p_value_vs_chance": 0.005,
                            "mean_image_share_in_hotspots": 0.038, "mean_error_share_in_hotspots": 0.205}},
        "error_detection": {
            "methods": {k: {"auroc": _interval(0.7), "average_precision": _interval(0.05)}
                        for k in ("som_cell", "kmeans_cell", "knn_impurity", "max_softmax")},
            "bootstrap": {"knn_impurity": {"point": 0.87, "lower": 0.83, "upper": 0.92},
                          "max_softmax": {"point": 0.98, "lower": 0.97, "upper": 0.99}}},
    }


def test_table_has_one_row_per_method_and_derives_labels_from_settings() -> None:
    table = latex_table(_results())
    assert "SOM 15$\\times$15" in table and "225 clusters" in table and "$k=15$" in table
    assert table.count("\\hline") == 6
    assert "0.980 (0.970--0.990)" in table
    assert "RQ" not in table


def test_summary_csv_is_written(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    write_summary_csv(path, _results())
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["method"] for r in rows] == ["som_cell", "kmeans_cell", "knn_impurity", "max_softmax"]


def test_report_outputs_are_written_next_to_the_results(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    write_report_outputs(_results(), tmp_path / "run")
    table = tmp_path / "run" / "error_pattern_baselines.tex"
    assert table.read_text(encoding="utf-8").startswith("\\begin{table}")
    assert (tmp_path / "run" / "summary.csv").exists()
