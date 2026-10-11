"""Week 3 figures for the selected SOM (SOM Setup with NNSOM).

Draws the figures that describe the SOM itself: topology, neuron
connections, U-Matrix, train/validation hit histograms, component planes,
feature maps and (with the extended switch) dense NNSOM plots.

Output (folders come from the SOM config):
    outputs/v1_mnist/p2_som_training_visualization/som_nnsom_training/figures/<model_name>/
        core, feature maps, component planes, extended, visualization_manifest.json
Log: som_nnsom_training/logs/<model_name>_visualize.log

Week 4 figures: cluster_analysis/cluster_figures.py.
Week 5 figures: error_pattern_analysis/error_figures.py.

Usage (from the code root):
  python src/v1_mnist/component/p2_som_training_visualization/som_nnsom_training/visualize_som.py \
      [--config CONFIG] [--model-name RUN_ID] [--extended]
"""

from __future__ import annotations

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

from src.v1_mnist.component.utils.dataset_info import dataset_info  # noqa: E402
from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402
from src.v1_mnist.component.utils.som_plotting import (  # noqa: E402
    build_data_dict,
    save_nnsom_plot,
    top_variance_features,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.extended_plots import (  # noqa: E402
    attempt_extended_plots,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.som_visualizations import (  # noqa: E402
    save_component_plane_pages,
    save_feature_hist_maps,
    save_native_component_planes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 3 NNSOM figures for the selected SOM."
    )
    add_common_args(parser)
    parser.add_argument(
        "--extended",
        action="store_true",
        help=(
            "Attempt additional dense professor-example plots "
            "(hist/box/violin/scatter/weights). "
            "These can be difficult to read on a 20x20 map."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, model_name = load_config_and_model_name(args.config, args.model_name, REPO_ROOT)

    output_dir = resolve_path(config.paths.figures_dir, REPO_ROOT) / model_name
    logs_dir = resolve_path(config.paths.logs_dir, REPO_ROOT)
    logger = setup_logger("visualize_som", log_file=logs_dir / f"{model_name}_visualize.log")

    core_dir = output_dir / "core"
    feature_dir = output_dir / "feature_maps"
    component_dir = output_dir / "component_planes"
    extended_dir = output_dir / "extended"
    make_figure_dirs(core_dir, feature_dir, component_dir, extended_dir)

    inputs = load_som_and_data(config, model_name, REPO_ROOT)
    som, data, model_metrics = inputs.som, inputs.data, inputs.model_metrics
    info = dataset_info(config)
    layer = info.feature_layer                       # e.g. "fc2"
    feature_dim = int(data.train.features.shape[1])  # equals config model.feature_dim (checked on load)

    train_dict, _ = build_data_dict(som, data.train.features, data.train.labels)
    val_dict, _ = build_data_dict(som, data.val.features, data.val.labels)
    # The extended plots use the split named by visualization.class_map_split.
    class_split_name, class_dict = class_map_dict(config, train_dict, val_dict)

    scaled_train = data.scaler.transform(data.train.features)
    feature_indices = top_variance_features(
        scaled_train,
        n_features=int(config.visualization.top_embedding_features),
    )

    generated = []
    failed = {}

    logger.info("=" * 72)
    logger.info("%s SOM Visualizations: Week 3", info.display_name)
    logger.info("=" * 72)
    logger.info("Model: %s", model_name)
    logger.info(
        "Grid:  %dx%d",
        model_metrics["som"]["grid_height"],
        model_metrics["som"]["grid_width"],
    )
    logger.info("Output: %s", output_dir)
    logger.info("Top-variance %s dimensions: %s", layer, feature_indices)

    if bool(
        config.visualization.generate_topology
    ):
        for plot_type, filename, title in [
            (
                "top",
                "topology",
                "SOM Topology",
            ),
            (
                "top_num",
                "topology_numbered",
                "SOM Topology with Neuron IDs",
            ),
        ]:
            try:
                generated.extend(
                    str(p)
                    for p in save_nnsom_plot(
                        som,
                        plot_type,
                        core_dir
                        / filename,
                        title=title,
                    )
                )
            except Exception as exc:
                failed[
                    plot_type
                ] = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    if bool(
        config.visualization.generate_neuron_connection
    ):
        try:
            generated.extend(
                str(p)
                for p in save_nnsom_plot(
                    som,
                    "neuron_connection",
                    core_dir
                    / "neuron_connection",
                    title=(
                        "SOM Neuron "
                        "Connection Map"
                    ),
                )
            )
        except Exception as exc:
            failed[
                "neuron_connection"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    if bool(
        config.visualization.generate_u_matrix
    ):
        try:
            generated.extend(
                str(p)
                for p in save_nnsom_plot(
                    som,
                    "neuron_dist",
                    core_dir
                    / "u_matrix_neuron_distance",
                    title=(
                        "SOM Neuron Distance Map "
                        "(U-Matrix)"
                    ),
                )
            )
        except Exception as exc:
            failed[
                "neuron_dist"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    if bool(
        config.visualization.generate_hit_histograms
    ):
        for (
            split_name,
            split_dict,
        ) in [
            (
                "train",
                train_dict,
            ),
            (
                "validation",
                val_dict,
            ),
        ]:
            try:
                generated.extend(
                    str(p)
                    for p in save_nnsom_plot(
                        som,
                        "hit_hist",
                        core_dir
                        / (
                            f"{split_name}_"
                            "hit_histogram"
                        ),
                        data_dict=split_dict,
                        title=(
                            split_name.capitalize()
                            + " Hit Histogram"
                        ),
                    )
                )
            except Exception as exc:
                failed[
                    f"hit_hist_{split_name}"
                ] = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    if bool(
        config.visualization.generate_component_planes
    ):
        try:
            generated.extend(
                str(p)
                for p in save_native_component_planes(
                    som,
                    train_dict,
                    component_dir
                    / f"component_planes_all{feature_dim}_nnsom",
                    feature_prefix=layer,
                    model_label=info.model_display_name,
                )
            )
        except Exception as exc:
            failed[
                "component_planes_native"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        try:
            generated.extend(
                str(p)
                for p in save_component_plane_pages(
                    som,
                    component_dir,
                    feature_indices=list(
                        range(feature_dim)
                    ),
                    features_per_page=12,
                    feature_prefix=layer,
                )
            )
        except Exception as exc:
            failed[
                "component_planes_paginated"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    try:
        generated.extend(
            str(p)
            for p in save_feature_hist_maps(
                som,
                train_dict,
                feature_indices,
                feature_dir,
                include_color=bool(
                    config.visualization.generate_feature_color_histograms
                ),
                include_gray=bool(
                    config.visualization.generate_feature_gray_histograms
                ),
                feature_prefix=layer,
                model_label=info.model_display_name,
            )
        )
    except Exception as exc:
        failed[
            "feature_hist_maps"
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
            extended_generated, extended_failed = attempt_extended_plots(
                som,
                class_dict,
                feature_indices,
                extended_dir,
                layer=layer,
                feature_dim=feature_dim,
            )
        generated.extend(extended_generated)
        failed.update(
            {
                "extended_" + key: value
                for key, value in extended_failed.items()
            }
        )

    manifest_path = write_manifest(
        output_dir,
        figures="som",
        model_name=model_name,
        inputs=inputs,
        generated=generated,
        failed=failed,
        repo_root=REPO_ROOT,
        extra={
            "class_map_split": class_split_name,
            f"top_variance_{layer}_features": feature_indices,
            "intentionally_not_default": {
                "hist_box_violin_scatter": (
                    f"{feature_dim} {layer} dimensions are abstract; "
                    "default output uses selected feature "
                    "hist maps and component planes instead."
                ),
                "weight_as_image": (
                    f"{feature_dim} CNN embedding dimensions have no "
                    "natural 2-D pixel arrangement, so "
                    f"reshaping {feature_dim} weights into an image "
                    "would be arbitrary."
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
