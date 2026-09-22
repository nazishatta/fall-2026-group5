"""Generate NNSOM visualizations for the selected MNIST SOM.

Default output:
    outputs/v1_mnist/som/figures/<model_name>/

The default set favors plots that remain scientifically meaningful and
readable for a SOM trained on abstract CNN embedding dimensions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from NNSOM.plots import SOMPlots

import matplotlib.cm as mpl_cm
from matplotlib import colormaps

# Compatibility shim for NNSOM + modern Matplotlib.
# NNSOM 1.8.x still calls matplotlib.cm.get_cmap(),
# which is unavailable in newer Matplotlib versions.
if not hasattr(mpl_cm, "get_cmap"):
    def _compat_get_cmap(name=None, lut=None):
        cmap = colormaps.get_cmap(name)
        if lut is not None:
            cmap = cmap.resampled(lut)
        return cmap
    mpl_cm.get_cmap = _compat_get_cmap



REPO_ROOT = Path(__file__).resolve().parents[4]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.v1_mnist.component.som.data import load_som_data
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.visualization.som_visualizations import (
    build_data_dict,
    compute_neuron_classification_stats,
    save_class_distribution_maps,
    save_complex_error_map,
    save_component_plane_pages,
    save_feature_hist_maps,
    save_native_component_planes,
    save_nnsom_plot,
    save_simple_grid_map,
    top_variance_features,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate NNSOM visualizations "
            "for the selected MNIST SOM."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="src/v1_mnist/component/configs/som.yaml",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Saved SOM filename/run-id. "
            "Defaults to visualization.selected_model."
        ),
    )

    parser.add_argument(
        "--extended",
        action="store_true",
        help=(
            "Attempt additional dense professor-example plots "
            "(pie/stem/hist/box/violin/scatter/weights). "
            "These can be difficult to read on a 20x20 map."
        ),
    )

    return parser.parse_args()


def load_model_and_grid(
    model_name: str,
    models_dir: Path,
    logs_dir: Path,
):
    """Load a saved SOM, inferring grid dimensions from its metrics record."""

    metrics_path = (
        logs_dir
        / f"{model_name}_metrics.json"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            "Could not infer SOM grid because the metrics "
            f"file is missing: {metrics_path}"
        )

    metrics = json.loads(
        metrics_path.read_text(
            encoding="utf-8"
        )
    )

    grid_height = int(
        metrics["som"]["grid_height"]
    )

    grid_width = int(
        metrics["som"]["grid_width"]
    )

    model_path = (
        models_dir
        / model_name
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            model_path
        )

    som = SOMPlots(
        dimensions=(
            grid_height,
            grid_width,
        )
    )

    loaded = som.load_pickle(
        model_path.name,
        str(
            model_path.parent
        )
        + os.sep,
    )

    if (
        loaded is not None
        and hasattr(
            loaded,
            "cluster_data",
        )
    ):
        som = loaded

    return som, metrics


def attempt_extended_plots(
    som,
    data_dict: dict,
    feature_indices: list[int],
    output_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    """Attempt dense NNSOM plots, recording any failures instead of aborting."""

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

    if len(feature_indices) < 2:
        return generated, failed

    f0 = int(
        feature_indices[0]
    )
    f1 = int(
        feature_indices[1]
    )

    jobs = [
        (
            "pie",
            {},
            "class_pie",
        ),
        (
            "stem",
            {},
            "class_stem",
        ),
        (
            "scatter",
            {
                "ind": [
                    f0,
                    f1,
                ],
                "reg_line": False,
            },
            (
                "scatter_fc2_"
                f"{f0:02d}_"
                f"{f1:02d}"
            ),
        ),
    ]

    for (
        plot_type,
        kwargs,
        filename,
    ) in jobs:
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
                **kwargs,
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

    for feature_id in feature_indices:
        feature_id = int(
            feature_id
        )

        try:
            paths = save_nnsom_plot(
                som,
                "hist",
                output_dir
                / (
                    f"hist_fc2_"
                    f"{feature_id:02d}"
                ),
                data_dict=extended_data_dict,
                ind=feature_id,
                title=(
                    "Cluster Histogram — "
                    f"fc2 Feature {feature_id}"
                ),
            )

            generated.extend(
                str(p)
                for p in paths
            )

        except Exception as exc:
            plt.close(
                "all"
            )

            failed[
                (
                    "hist_fc2_"
                    f"{feature_id:02d}"
                )
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    try:
        paths = save_nnsom_plot(
            som,
            "box",
            output_dir
            / "box_top_fc2_features",
            data_dict=extended_data_dict,
            ind=[
                int(i)
                for i in feature_indices
            ],
            title=(
                "Box Plot — "
                "Top-Variance fc2 Features"
            ),
        )

        generated.extend(
            str(p)
            for p in paths
        )

    except Exception as exc:
        plt.close(
            "all"
        )

        failed[
            "box_top_fc2_features"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    try:
        paths = save_nnsom_plot(
            som,
            "violin",
            output_dir
            / "violin_top_fc2_features",
            data_dict=extended_data_dict,
            ind=[
                int(i)
                for i in feature_indices
            ],
            title=(
                "Violin Plot — "
                "Top-Variance fc2 Features"
            ),
        )

        generated.extend(
            str(p)
            for p in paths
        )

    except Exception as exc:
        plt.close(
            "all"
        )

        failed[
            "violin_top_fc2_features"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    # Weight-line plot does not require data_dict.
    try:
        paths = save_nnsom_plot(
            som,
            "wgts",
            output_dir
            / "weights_line_plot",
            title=(
                "SOM Weight Line Plot "
                "(84 fc2 dimensions)"
            ),
            dpi=180,
        )

        generated.extend(
            str(p)
            for p in paths
        )

    except Exception as exc:
        plt.close("all")
        failed["wgts"] = (
            f"{type(exc).__name__}: {exc}"
        )

    # component_positions is valid but only projects the first
    # two embedding weight dimensions, so treat it as extended.
    try:
        paths = save_nnsom_plot(
            som,
            "component_positions",
            output_dir
            / "component_positions_first_two",
            data_dict=extended_data_dict,
            title=(
                "SOM Weight Positions "
                "(first two fc2 dimensions)"
            ),
        )

        generated.extend(
            str(p)
            for p in paths
        )

    except Exception as exc:
        plt.close("all")
        failed[
            "component_positions"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

    return generated, failed


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = (REPO_ROOT / config_path).resolve()

    config = load_config(str(config_path))

    model_name = (
        args.model_name
        or config.visualization.selected_model
    )

    def resolve_path(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    models_dir = resolve_path(config.paths.som_models_dir)
    logs_dir = resolve_path(config.paths.logs_dir)
    figures_dir = resolve_path(config.paths.figures_dir)
    embeddings_dir = config.paths.embeddings_dir

    output_dir = (
        figures_dir
        / model_name
    )

    core_dir = (
        output_dir
        / "core"
    )

    class_dir = (
        output_dir
        / "class_maps"
    )

    feature_dir = (
        output_dir
        / "feature_maps"
    )

    component_dir = (
        output_dir
        / "component_planes"
    )

    extended_dir = (
        output_dir
        / "extended"
    )

    analysis_map_dir = (
        output_dir
        / "analysis_maps"
    )

    for directory in (
        core_dir,
        class_dir,
        feature_dir,
        component_dir,
        extended_dir,
        analysis_map_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    data = load_som_data(
        embeddings_dir=embeddings_dir,
        expected_feature_dim=84,
    )

    som, model_metrics = (
        load_model_and_grid(
            model_name=model_name,
            models_dir=models_dir,
            logs_dir=logs_dir,
        )
    )

    # Reattach the train-fitted scaler explicitly.
    # This is safer than relying on a pickled bound transform method.
    som.norm_func = (
        data.scaler.transform
    )

    train_dict, _ = (
        build_data_dict(
            som,
            data.train.features,
            data.train.labels,
        )
    )

    val_dict, val_clusters = (
        build_data_dict(
            som,
            data.val.features,
            data.val.labels,
        )
    )

    val_stats = (
        compute_neuron_classification_stats(
            clusters=val_clusters,
            labels=data.val.labels,
            preds=data.val.preds,
        )
    )

    class_split_name = str(
        config.visualization.class_map_split
    )

    if class_split_name == "train":
        class_dict = train_dict
    elif class_split_name == "val":
        class_dict = val_dict
    else:
        raise ValueError(
            "visualization.class_map_split "
            "must be 'train' or 'val'."
        )

    scaled_train = (
        data.scaler.transform(
            data.train.features
        )
    )

    n_top_features = int(
        config.visualization.top_embedding_features
    )

    feature_indices = (
        top_variance_features(
            scaled_train,
            n_features=n_top_features,
        )
    )

    generated = []
    failed = {}

    print("=" * 72)
    print("MNIST v1 SOM Visualizations")
    print("=" * 72)
    print(
        f"Model: {model_name}"
    )
    print(
        "Grid:  "
        f"{model_metrics['som']['grid_height']}x"
        f"{model_metrics['som']['grid_width']}"
    )
    print(
        f"Output: {output_dir}"
    )
    print(
        "Top-variance fc2 dimensions: "
        f"{feature_indices}"
    )

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
                        dpi=220,
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
                    / "component_planes_all84_nnsom",
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
                        range(84)
                    ),
                    features_per_page=12,
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
            for p in save_class_distribution_maps(
                som,
                class_dict,
                class_dir,
                num_classes=10,
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
            )
        )
    except Exception as exc:
        failed[
            "feature_hist_maps"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

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
                        analysis_map_dir
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
                        analysis_map_dir
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
                    analysis_map_dir
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

            (
                extended_generated,
                extended_failed,
            ) = attempt_extended_plots(
                som,
                class_dict,
                feature_indices,
                extended_dir,
            )

        generated.extend(
            extended_generated
        )

        failed.update(
            {
                (
                    "extended_"
                    + key
                ): value
                for (
                    key,
                    value
                ) in extended_failed.items()
            }
        )

    manifest = {
        "model_name": model_name,
        "grid": [
            int(
                model_metrics[
                    "som"
                ][
                    "grid_height"
                ]
            ),
            int(
                model_metrics[
                    "som"
                ][
                    "grid_width"
                ]
            ),
        ],
        "train_shape": list(
            data.train.features.shape
        ),
        "validation_shape": list(
            data.val.features.shape
        ),
        "feature_dim": int(
            data.train.features.shape[1]
        ),
        "class_map_split": (
            class_split_name
        ),
        "top_variance_fc2_features": (
            feature_indices
        ),
        "generated_files": [
            str(
                Path(path).relative_to(
                    REPO_ROOT
                )
            )
            if Path(path).is_absolute()
            else path
            for path in generated
        ],
        "failed_or_skipped_due_to_runtime": (
            failed
        ),
        "intentionally_not_default": {
            "pie": (
                "Per-neuron pie charts are visually dense "
                "for the selected SOM and are not part of the core analysis."
            ),
            "stem": (
                "Per-neuron 10-class stem plots are "
                "visually dense and are not part of the core analysis."
            ),
            "hist_box_violin_scatter": (
                "84 fc2 dimensions are abstract; "
                "default output uses selected feature "
                "hist maps and component planes instead."
            ),
            "weight_as_image": (
                "84 CNN embedding dimensions have no "
                "natural 2-D pixel arrangement, so "
                "reshaping 84 weights into an image "
                "would be arbitrary."
            ),
        },
        "test_evaluated": False,
    }

    manifest_path = (
        output_dir
        / "visualization_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"Generated {len(generated)} figure file(s)."
    )
    print(
        f"Manifest: {manifest_path}"
    )

    if failed:
        print(
            "\nSome optional/individual plots failed:"
        )

        for name, message in failed.items():
            print(
                f"  {name}: {message}"
            )

    print(
        "\nTest split was NOT used."
    )


if __name__ == "__main__":
    main()
