"""Week 5 figures for the selected SOM (Error Pattern Analysis).

Draws the error maps on the validation split: the error-rate map
(colour: error rate, size: support) and the complex error map.

Output (folders come from the SOM config):
    outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/figures/<model_name>/
        analysis_maps, visualization_manifest.json
Log: error_pattern_analysis/logs/<model_name>_error_figures.log

Usage (from the code root):
  python src/v1_mnist/component/p2_som_training_visualization/error_pattern_analysis/error_figures.py \
      [--config CONFIG] [--model-name RUN_ID]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.v1_mnist.component.utils.som_types import SOMModel

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# som_figure_setup first: it selects the Agg backend and patches NNSOM for new Matplotlib.
from src.v1_mnist.component.utils.som_figure_setup import (  # noqa: E402
    add_common_args,
    load_config_and_model_name,
    load_som_and_data,
    make_figure_dirs,
    resolve_path,
    write_manifest,
)
import numpy as np  # noqa: E402

from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402
from src.v1_mnist.component.utils.som_plotting import (  # noqa: E402
    build_data_dict,
    save_nnsom_plot,
)
from src.v1_mnist.component.utils.som_neuron_maps import (  # noqa: E402
    compute_neuron_classification_stats,
    save_simple_grid_map,
)


def save_complex_error_map(
    som: SOMModel,
    data_dict: dict,
    stats: dict[str, np.ndarray],
    output_path: Path,
) -> list[Path]:
    """Save a multiclass complex-hit map (face: dominant true class, edge: dominant wrong prediction)."""

    face_labels = np.asarray(
        stats[
            "dominant_true_class"
        ],
        dtype=np.float64,
    )

    edge_labels = np.asarray(
        stats[
            "dominant_wrong_prediction"
        ],
        dtype=np.float64,
    )

    edge_width = np.asarray(
        stats[
            "edge_width"
        ],
        dtype=np.float64,
    )

    # Empty neurons may contain NaN labels.
    # Give them valid placeholder labels; their support is zero.
    face_labels = np.nan_to_num(
        face_labels,
        nan=0.0,
    )

    edge_labels = np.nan_to_num(
        edge_labels,
        nan=0.0,
    )

    plot_dict = dict(
        data_dict
    )

    # NNSOM plot("complex_hist") expects:
    # column 0 = face label
    # column 1 = edge label
    # column 2 = edge width
    plot_dict[
        "add_2d_array"
    ] = np.column_stack(
        [
            face_labels,
            edge_labels,
            edge_width,
        ]
    )

    return save_nnsom_plot(
        som,
        "complex_hist",
        output_path,
        data_dict=plot_dict,
        use_add_array=True,
        title=(
            "Validation Error Geography — "
            "Face: Dominant True Class; "
            "Edge: Dominant Wrong Prediction; "
            "Width: Error Concentration"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 5 error-pattern figures for the selected SOM."
    )
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, model_name = load_config_and_model_name(args.config, args.model_name, REPO_ROOT)

    error_dir = resolve_path(config.paths.error_dir, REPO_ROOT)
    error_fig_dir = error_dir / "figures" / model_name
    logger = setup_logger(
        "error_figures",
        log_file=error_dir / "logs" / f"{model_name}_error_figures.log",
    )

    error_analysis_map_dir = error_fig_dir / "analysis_maps"
    make_figure_dirs(error_analysis_map_dir)

    inputs = load_som_and_data(config, model_name, REPO_ROOT)
    som, data, model_metrics = inputs.som, inputs.data, inputs.model_metrics

    val_dict, val_clusters = build_data_dict(som, data.val.features, data.val.labels)
    val_stats = compute_neuron_classification_stats(
        clusters=val_clusters,
        labels=data.val.labels,
        preds=data.val.preds,
    )

    generated = []
    failed = {}

    logger.info("=" * 72)
    logger.info("SOM Visualizations: Week 5 (error pattern analysis)")
    logger.info("=" * 72)
    logger.info("Model: %s", model_name)
    logger.info(
        "Grid:  %dx%d",
        model_metrics["som"]["grid_height"],
        model_metrics["som"]["grid_width"],
    )
    logger.info("Output: %s", error_fig_dir)

    if bool(
        config.visualization
        .generate_simple_grid_error_support
    ):
        try:
            generated.extend(
                str(p)
                for p in save_simple_grid_map(
                    som,
                    val_dict,
                    values=val_stats[
                        "error_rate"
                    ],
                    sizes=val_stats[
                        "support"
                    ],
                    output_path=(
                        error_analysis_map_dir
                        / "validation_error_rate_support"
                    ),
                    title=(
                        "Validation Error Geography — "
                        "Color: Error Rate; "
                        "Size: Support"
                    ),
                )
            )

        except Exception as exc:
            failed[
                "simple_grid_error_support"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    if bool(
        config.visualization
        .generate_complex_error_map
    ):
        try:
            generated.extend(
                str(p)
                for p in save_complex_error_map(
                    som,
                    val_dict,
                    val_stats,
                    error_analysis_map_dir
                    / "validation_complex_error_map",
                )
            )

        except Exception as exc:
            failed[
                "complex_error_map"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )


    manifest_path = write_manifest(
        error_fig_dir,
        figures="error",
        model_name=model_name,
        inputs=inputs,
        generated=generated,
        failed=failed,
        repo_root=REPO_ROOT,
    )

    logger.info("Generated %d figure file(s).", len(generated))
    logger.info("Manifest: %s", manifest_path)
    if failed:
        logger.warning("Some optional/individual plots failed:")
        for name, message in failed.items():
            logger.warning("  %s: %s", name, message)
    logger.info("Test split was NOT used.")


if __name__ == "__main__":
    main()
