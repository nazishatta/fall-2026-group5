"""Generate an interactive SOM neighborhood-radius schedule plot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import plotly.graph_objects as go


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.analysis.run_settings import (  # noqa: E402
    DEFAULT_CONFIG,
    selected_model_from_config,
)

from src.v1_mnist.component.interactive.som_interactive_plots_2.artifacts import (  # noqa: E402
    load_interactive_artifacts,
)

# Set in configure() from the SOM config (default: the 15x15 headline model).
METRICS_PATH: Path = Path()
QE_HISTORY_PATH: Path = Path()
OUTPUT_PATH: Path = Path()
OUTPUT_NAME = "som_neighborhood_radius.html"


def configure(config_path=None) -> None:
    """Point this script at the selected SOM's metrics and QE history."""
    global METRICS_PATH, QE_HISTORY_PATH, OUTPUT_PATH
    run_id = selected_model_from_config(config_path, REPO_ROOT)
    METRICS_PATH = REPO_ROOT / "outputs/v1_mnist/som/logs" / f"{run_id}_metrics.json"
    QE_HISTORY_PATH = REPO_ROOT / "outputs/v1_mnist/som/tables" / f"{run_id}_qe_history.csv"
    OUTPUT_PATH = REPO_ROOT / "demo/fig/v1_mnist/som_interactive_plots" / f"{run_id}_{OUTPUT_NAME}"


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

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="SOM config naming the selected model.")
    configure(parser.parse_args().config)

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
