"""Week 4 figures for the selected SOM (Cluster Analysis).

Draws the class figures: one class-distribution map per class (colour and
grey; for MNIST, per digit), the class-purity map (colour: purity, size: support) and, with the
extended switch, the per-neuron class pie and stem plots.

Output (folders come from the SOM config):
    outputs/v1_mnist/p2_som_training_visualization/cluster_analysis/figures/<model_name>/
        class_maps, analysis_maps, extended, visualization_manifest.json
Log: cluster_analysis/logs/<model_name>_cluster_figures.log

Usage (from the code root):
  python src/v1_mnist/component/p2_som_training_visualization/cluster_analysis/cluster_figures.py \
      [--config CONFIG] [--model-name RUN_ID] [--extended]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.v1_mnist.component.utils.som_types import SOMModel

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# som_figure_setup first: it selects the Agg backend and patches NNSOM for new Matplotlib.
from src.v1_mnist.component.utils.som_figure_setup import (  # noqa: E402
    add_common_args,
    class_map_dict,
    load_config_and_model_name,
    load_som_and_data,
    make_figure_dirs,
    resolve_path,
    write_manifest,
)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.v1_mnist.component.utils.dataset_info import dataset_info  # noqa: E402
from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402
from src.v1_mnist.component.utils.som_plotting import (  # noqa: E402
    build_data_dict,
    save_nnsom_plot,
)
from src.v1_mnist.component.utils.som_neuron_maps import (  # noqa: E402
    compute_neuron_classification_stats,
    save_simple_grid_map,
)


def save_class_distribution_maps(
    som: SOMModel,
    data_dict: dict,
    output_dir: Path,
    *,
    num_classes: int,
    display_name: str,
    class_label: str,
    include_color: bool = True,
    include_gray: bool = True,
) -> list[Path]:
    """Generate one NNSOM class-distribution map per class (file names use class_label, e.g. digit_0)."""

    saved = []

    for digit in range(
        num_classes
    ):
        if include_color:
            saved.extend(
                save_nnsom_plot(
                    som,
                    "color_hist",
                    (
                        output_dir
                        / f"{class_label}_{digit}_color_hist"
                    ),
                    data_dict=data_dict,
                    title=(
                        f"{display_name} {class_label.title()} "
                        f"{digit} Distribution"
                    ),
                    target_class=digit,
                )
            )

        if include_gray:
            saved.extend(
                save_nnsom_plot(
                    som,
                    "gray_hist",
                    (
                        output_dir
                        / f"{class_label}_{digit}_gray_hist"
                    ),
                    data_dict=data_dict,
                    title=(
                        f"{display_name} {class_label.title()} "
                        f"{digit} Distribution "
                        "(Gray Hist)"
                    ),
                    target_class=digit,
                )
            )

    return saved


def attempt_class_composition_plots(
    som: SOMModel,
    data_dict: dict,
    output_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    """Per-neuron class pie and stem plots (dense), recording failures instead of aborting."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = []
    failed = {}

    # Prepare a float64 copy of the data for extended plots to avoid CuPy/float32 issues.
    # This does not modify the original embeddings used elsewhere.
    extended_data_dict = dict(data_dict)
    if "data" in extended_data_dict:
        extended_data_dict["data"] = np.asarray(
            extended_data_dict["data"], dtype=np.float64
        )

    for plot_type, filename in (("pie", "class_pie"), ("stem", "class_stem")):
        try:
            paths = save_nnsom_plot(
                som,
                plot_type,
                output_dir / filename,
                data_dict=extended_data_dict,
                title=(
                    "Extended NNSOM "
                    f"{plot_type}"
                ),
            )

            generated.extend(
                str(p)
                for p in paths
            )

        except Exception as exc:
            plt.close("all")

            failed[
                plot_type
            ] = (
                f"{type(exc).__name__}: {exc}"
            )

    return generated, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 4 cluster-analysis figures for the selected SOM."
    )
    add_common_args(parser)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Also draw the dense per-neuron class pie and stem plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, model_name = load_config_and_model_name(args.config, args.model_name, REPO_ROOT)

    cluster_dir = resolve_path(config.paths.cluster_dir, REPO_ROOT)
    cluster_fig_dir = cluster_dir / "figures" / model_name
    logger = setup_logger(
        "cluster_figures",
        log_file=cluster_dir / "logs" / f"{model_name}_cluster_figures.log",
    )

    class_dir = cluster_fig_dir / "class_maps"
    cluster_extended_dir = cluster_fig_dir / "extended"
    cluster_analysis_map_dir = cluster_fig_dir / "analysis_maps"
    make_figure_dirs(class_dir, cluster_extended_dir, cluster_analysis_map_dir)

    info = dataset_info(config)
    inputs = load_som_and_data(config, model_name, REPO_ROOT)
    som, data, model_metrics = inputs.som, inputs.data, inputs.model_metrics

    train_dict, _ = build_data_dict(som, data.train.features, data.train.labels)
    val_dict, val_clusters = build_data_dict(som, data.val.features, data.val.labels)
    val_stats = compute_neuron_classification_stats(
        clusters=val_clusters,
        labels=data.val.labels,
        preds=data.val.preds,
    )
    class_split_name, class_dict = class_map_dict(config, train_dict, val_dict)

    generated = []
    failed = {}

    logger.info("=" * 72)
    logger.info("%s SOM Visualizations: Week 4 (cluster analysis)", info.display_name)
    logger.info("=" * 72)
    logger.info("Model: %s", model_name)
    logger.info(
        "Grid:  %dx%d",
        model_metrics["som"]["grid_height"],
        model_metrics["som"]["grid_width"],
    )
    logger.info("Output: %s", cluster_fig_dir)

    try:
        generated.extend(
            str(p)
            for p in save_class_distribution_maps(
                som,
                class_dict,
                class_dir,
                num_classes=info.num_classes,
                display_name=info.display_name,
                class_label=info.class_label,
                include_color=bool(
                    config.visualization.generate_class_color_histograms
                ),
                include_gray=bool(
                    config.visualization.generate_class_gray_histograms
                ),
            )
        )
    except Exception as exc:
        failed[
            "class_distribution_maps"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    if bool(
        config.visualization
        .generate_simple_grid_purity_support
    ):
        try:
            generated.extend(
                str(p)
                for p in save_simple_grid_map(
                    som,
                    val_dict,
                    values=val_stats[
                        "purity"
                    ],
                    sizes=val_stats[
                        "support"
                    ],
                    output_path=(
                        cluster_analysis_map_dir
                        / "validation_purity_support"
                    ),
                    title=(
                        "Validation Representation Structure — "
                        "Color: Class Purity; "
                        "Size: Support"
                    ),
                )
            )

        except Exception as exc:
            failed[
                "simple_grid_purity_support"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )


    extended_requested = (
        bool(
            config.visualization.generate_extended_dense_plots
        )
        or args.extended
    )

    if extended_requested:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )
            extended_generated, extended_failed = attempt_class_composition_plots(
                som,
                class_dict,
                cluster_extended_dir,
            )
        generated.extend(extended_generated)
        failed.update(
            {
                "extended_" + key: value
                for key, value in extended_failed.items()
            }
        )

    manifest_path = write_manifest(
        cluster_fig_dir,
        figures="cluster",
        model_name=model_name,
        inputs=inputs,
        generated=generated,
        failed=failed,
        repo_root=REPO_ROOT,
        extra={
            "class_map_split": class_split_name,
            "intentionally_not_default": {
                "pie": (
                    "Per-neuron pie charts are visually dense "
                    "for the selected SOM and are not part of the core analysis."
                ),
                "stem": (
                    "Per-neuron 10-class stem plots are "
                    "visually dense and are not part of the core analysis."
                ),
            },
        },
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
