"""Generate an interactive train-vs-validation SOM quality summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[4]

METRICS_PATH = (
    REPO_ROOT
    / "outputs/v1_mnist/som/logs/"
    "som_15x15_seed42_final_metrics.json"
)

OUTPUT_PATH = (
    REPO_ROOT
    / "demo/fig/v1_mnist/som_interactive_plots/"
    "som_train_validation_quality.html"
)


def load_metrics() -> dict[str, Any]:
    """Load and validate the frozen SOM metrics."""

    if not METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"Required SOM metrics not found: {METRICS_PATH}"
        )

    metrics = json.loads(
        METRICS_PATH.read_text(encoding="utf-8")
    )

    if metrics.get("test_evaluated") is not False:
        raise ValueError(
            "Development visualization requires "
            "test_evaluated=false."
        )

    return metrics


def build_quality_figure() -> go.Figure:
    """Build the interactive train-vs-validation quality summary."""

    metrics = load_metrics()

    experiment = metrics["experiment"]
    som = metrics["som"]
    train = metrics["train_metrics"]
    validation = metrics["validation_metrics"]

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Quantization Error",
            "1st-Order Topological Error",
            "Mean Hits per Occupied Neuron",
            "Maximum Hits per Neuron",
        ),
    )

    split_names = ["Train", "Validation"]

    comparisons = (
        (
            [
                train["quantization_error"],
                validation["quantization_error"],
            ],
            1,
            1,
            "QE",
        ),
        (
            [
                train["topological_error_1st_order_pct"],
                validation["topological_error_1st_order_pct"],
            ],
            1,
            2,
            "Topological error (%)",
        ),
        (
            [
                train["mean_hits_per_occupied_neuron"],
                validation["mean_hits_per_occupied_neuron"],
            ],
            2,
            1,
            "Mean hits",
        ),
        (
            [
                train["max_hits_per_neuron"],
                validation["max_hits_per_neuron"],
            ],
            2,
            2,
            "Maximum hits",
        ),
    )

    for values, row, col, label in comparisons:
        figure.add_trace(
            go.Bar(
                x=split_names,
                y=values,
                text=[
                    f"{float(value):.4f}"
                    for value in values
                ],
                textposition="auto",
                customdata=[
                    [
                        split_names[index],
                        float(value),
                        int(
                            train["occupied_neurons"]
                            if index == 0
                            else validation["occupied_neurons"]
                        ),
                    ]
                    for index, value in enumerate(values)
                ],
                hovertemplate=(
                    "Split: %{customdata[0]}<br>"
                    + label
                    + ": %{customdata[1]:.6f}<br>"
                    "Occupied neurons: %{customdata[2]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    figure.update_yaxes(
        title_text="Quantization error",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Error (%)",
        row=1,
        col=2,
    )
    figure.update_yaxes(
        title_text="Samples",
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title_text="Samples",
        row=2,
        col=2,
    )

    figure.update_layout(
        title=(
            "MNIST NNSOM Train vs Validation Quality Summary"
            f"<br><sup>{experiment['run_id']} | "
            f"{som['grid_height']}×{som['grid_width']} grid | "
            f"seed {som['random_seed']} | "
            "development splits only</sup>"
        ),
        template="plotly_white",
        height=750,
        hoverlabel={"align": "left"},
    )

    return figure


def main() -> None:
    """Generate and save the interactive HTML visualization."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = build_quality_figure()

    figure.write_html(
        OUTPUT_PATH,
        include_plotlyjs="cdn",
        full_html=True,
    )

    print(f"Saved interactive plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
