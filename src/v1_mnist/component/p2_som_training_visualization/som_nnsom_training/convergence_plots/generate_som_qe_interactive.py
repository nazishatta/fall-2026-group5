"""Generate an interactive SOM quantization-error convergence plot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import plotly.graph_objects as go


REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.logging import get_logger  # noqa: E402
from src.v1_mnist.component.utils.run_settings import (  # noqa: E402
    DEFAULT_CONFIG,
    som_run_folders,
)

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.convergence_plots.artifacts import (  # noqa: E402
    load_interactive_artifacts,
)

# Set in configure() from the SOM config (default: the 15x15 headline model).
METRICS_PATH: Path = Path()
QE_HISTORY_PATH: Path = Path()
OUTPUT_PATH: Path = Path()
OUTPUT_NAME = "som_qe_convergence.html"


def configure(config_path: str | Path | None = None, run_id: str | None = None) -> None:
    """Point this script at the selected SOM's metrics and QE history."""
    global METRICS_PATH, QE_HISTORY_PATH, OUTPUT_PATH
    folders = som_run_folders(config_path, REPO_ROOT, run_id)
    METRICS_PATH = folders.metrics_path
    QE_HISTORY_PATH = folders.qe_history_path
    OUTPUT_PATH = folders.convergence_plots_dir / f"{folders.run_id}_{OUTPUT_NAME}"


def build_qe_figure() -> go.Figure:
    """Build the interactive SOM QE convergence figure."""

    artifacts = load_interactive_artifacts(
        METRICS_PATH,
        QE_HISTORY_PATH,
    )

    epochs = [
        row["epoch"]
        for row in artifacts.qe_history
    ]
    qe_values = [
        row["quantization_error"]
        for row in artifacts.qe_history
    ]

    hover_text = [
        (
            f"Epoch: {int(row['epoch'])}<br>"
            f"Quantization error: "
            f"{row['quantization_error']:.6f}<br>"
            f"Neighborhood radius: "
            f"{row['neighborhood_radius']:.3f}"
        )
        for row in artifacts.qe_history
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=qe_values,
            mode="lines",
            name="Training QE",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    figure.add_hline(
        y=artifacts.train_qe,
        line_dash="dash",
        annotation_text=(
            f"Final train QE = {artifacts.train_qe:.4f}"
        ),
        annotation_position="top right",
    )

    figure.add_hline(
        y=artifacts.validation_qe,
        line_dash="dot",
        annotation_text=(
            "Validation QE = "
            f"{artifacts.validation_qe:.4f}"
        ),
        annotation_position="bottom right",
    )

    figure.update_layout(
        title=(
            "MNIST NNSOM Quantization Error Convergence"
            f"<br><sup>{artifacts.run_id} | "
            f"{artifacts.grid_height}×"
            f"{artifacts.grid_width} grid | "
            f"seed {artifacts.random_seed}</sup>"
        ),
        xaxis_title="Training epoch",
        yaxis_title="Quantization error",
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
    parser.add_argument("--run-id", default=None,
                        help="SOM run to chart (default: visualization.selected_model in the config).")
    args = parser.parse_args()
    configure(args.config, args.run_id)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = build_qe_figure()

    figure.write_html(
        OUTPUT_PATH,
        include_plotlyjs="cdn",
        full_html=True,
    )

    get_logger("v1_mnist.som_convergence_plots").info("Saved interactive plot: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
