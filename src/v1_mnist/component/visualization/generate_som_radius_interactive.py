"""Generate an interactive SOM neighborhood-radius schedule plot."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from src.v1_mnist.component.visualization.som_interactive import (
    load_interactive_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[4]

METRICS_PATH = (
    REPO_ROOT
    / "outputs/v1_mnist/som/logs/"
    "som_15x15_seed42_final_metrics.json"
)

QE_HISTORY_PATH = (
    REPO_ROOT
    / "outputs/v1_mnist/som/tables/"
    "som_15x15_seed42_final_qe_history.csv"
)

OUTPUT_PATH = (
    REPO_ROOT
    / "demo/fig/v1_mnist/som_interactive_plots/"
    "som_neighborhood_radius.html"
)


def build_radius_figure() -> go.Figure:
    """Build the interactive SOM neighborhood-radius figure."""

    artifacts = load_interactive_artifacts(
        METRICS_PATH,
        QE_HISTORY_PATH,
    )

    epochs = [
        row["epoch"]
        for row in artifacts.qe_history
    ]

    radii = [
        row["neighborhood_radius"]
        for row in artifacts.qe_history
    ]

    hover_text = [
        (
            f"Epoch: {int(row['epoch'])}<br>"
            f"Neighborhood radius: "
            f"{row['neighborhood_radius']:.3f}<br>"
            f"Quantization error: "
            f"{row['quantization_error']:.6f}"
        )
        for row in artifacts.qe_history
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=radii,
            mode="lines",
            name="Neighborhood radius",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    figure.update_layout(
        title=(
            "MNIST NNSOM Neighborhood Radius Schedule"
            f"<br><sup>{artifacts.run_id} | "
            f"{artifacts.grid_height}×"
            f"{artifacts.grid_width} grid | "
            f"seed {artifacts.random_seed}</sup>"
        ),
        xaxis_title="Training epoch",
        yaxis_title="Neighborhood radius",
        hovermode="x unified",
        template="plotly_white",
        legend_title="Series",
    )

    return figure


def main() -> None:
    """Generate and save the interactive HTML visualization."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = build_radius_figure()

    figure.write_html(
        OUTPUT_PATH,
        include_plotlyjs="cdn",
        full_html=True,
    )

    print(f"Saved interactive plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
