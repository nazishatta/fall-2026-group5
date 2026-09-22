"""Reusable SOM visualization helpers.

The functions intentionally use NNSOM's own plotting methods when they
are informative for MNIST and add paginated component-plane views so
that all 84 LeNet-fc2 embedding dimensions remain readable on the
selected SOM.

Important NNSOM behavior:
``hit_hist``, ``gray_hist`` and ``color_hist`` cache ``som.outputs``.
Before each such plot we force a fresh simulation so train and validation
plots cannot accidentally reuse assignments from a previous split.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from NNSOM.utils import (
    count_classes_in_cluster,
    get_edge_widths,
    get_hexagon_shape,
    get_ind_misclassified,
    get_perc_misclassified,
    majority_class_cluster,
)


def save_figure(
    fig,
    output_path: Path,
    *,
    dpi: int = 300,
    also_pdf: bool = False,
) -> list[Path]:
    """Save a Matplotlib figure and close it."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = output_path.with_suffix(
        ".png"
    )

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    saved = [png_path]

    if also_pdf:
        pdf_path = output_path.with_suffix(
            ".pdf"
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        saved.append(
            pdf_path
        )

    plt.close(fig)

    return saved


def build_data_dict(
    som,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict, list]:
    """Create the data_dict format used by NNSOM plotting examples."""

    (
        clusters,
        _distances,
        _max_distances,
        _sizes,
    ) = som.cluster_data(
        features
    )

    data_dict = {
        "data": features,
        "target": labels,
        "clust": clusters,
    }

    return data_dict, clusters


def _force_fresh_hit_simulation(
    som,
) -> None:
    """Prevent NNSOM hit-family plots from reusing stale cached outputs."""

    som.sim_flag = True
    som.outputs = None


def save_nnsom_plot(
    som,
    plot_type: str,
    output_path: Path,
    *,
    data_dict=None,
    title: str | None = None,
    dpi: int = 300,
    **kwargs,
) -> list[Path]:
    """Run an NNSOM plot and persist the resulting figure."""

    plt.close("all")

    if plot_type in {
        "hit_hist",
        "gray_hist",
        "color_hist",
        "complex_hist",
    }:
        _force_fresh_hit_simulation(
            som
        )

    result = som.plot(
        plot_type,
        data_dict,
        **kwargs,
    )

    if (
        isinstance(result, tuple)
        and len(result) > 0
        and hasattr(
            result[0],
            "savefig",
        )
    ):
        fig = result[0]
    else:
        # component_planes() currently calls plt.show()
        # and does not return the figure.
        fig = plt.gcf()

    if title:
        fig.suptitle(
            title,
            fontsize=14,
        )

    return save_figure(
        fig,
        output_path,
        dpi=dpi,
    )


def save_native_component_planes(
    som,
    data_dict: dict,
    output_path: Path,
    *,
    feature_prefix: str = "fc2",
) -> list[Path]:
    """Save NNSOM's native all-feature component-plane figure."""

    plt.close("all")

    som.plot(
        "component_planes",
        data_dict,
    )

    fig = plt.gcf()

    num_features = int(
        som.w.shape[1]
    )

    # NNSOM leaves the subplots untitled. Add compact feature labels.
    for feature_id, ax in enumerate(
        fig.axes[:num_features]
    ):
        ax.set_title(
            f"{feature_prefix}_{feature_id}",
            fontsize=5,
            pad=1,
        )

    for ax in fig.axes[
        num_features:
    ]:
        ax.set_visible(False)

    fig.suptitle(
        (
            "NNSOM Component Planes — "
            f"{num_features} LeNet fc2 Features"
        ),
        fontsize=12,
    )

    fig.set_size_inches(
        18,
        18,
    )

    return save_figure(
        fig,
        output_path,
        dpi=220,
    )


def save_component_plane_pages(
    som,
    output_dir: Path,
    *,
    feature_indices=None,
    features_per_page: int = 12,
    feature_prefix: str = "fc2",
) -> list[Path]:
    """Create readable paginated component planes for all/specified features.

    This reproduces the same information as NNSOM's component-plane plot:
    one map per embedding dimension, colored by the learned neuron weight.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights = np.asarray(
        som.w,
        dtype=np.float64,
    )

    positions = np.asarray(
        som.pos,
        dtype=np.float64,
    )

    if positions.shape[0] != 2:
        positions = positions.T

    num_neurons, num_features = (
        weights.shape
    )

    if feature_indices is None:
        feature_indices = list(
            range(num_features)
        )

    feature_indices = [
        int(i)
        for i in feature_indices
    ]

    shapex, shapey = (
        get_hexagon_shape()
    )

    columns = 4
    rows = math.ceil(
        features_per_page
        / columns
    )

    saved = []

    for page_start in range(
        0,
        len(feature_indices),
        features_per_page,
    ):
        page_features = feature_indices[
            page_start:
            page_start
            + features_per_page
        ]

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(
                4 * columns,
                3.5 * rows,
            ),
        )

        axes = np.atleast_1d(
            axes
        ).ravel()

        for ax in axes:
            ax.set_aspect(
                "equal"
            )
            ax.axis(
                "off"
            )

        for local_index, feature_id in enumerate(
            page_features
        ):
            ax = axes[
                local_index
            ]

            feature_weights = weights[
                :,
                feature_id,
            ]

            vmin = float(
                np.min(
                    feature_weights
                )
            )

            vmax = float(
                np.max(
                    feature_weights
                )
            )

            if np.isclose(
                vmin,
                vmax,
            ):
                vmax = vmin + 1e-12

            norm = mcolors.Normalize(
                vmin=vmin,
                vmax=vmax,
            )

            for neuron_id in range(
                num_neurons
            ):
                color = plt.cm.viridis(
                    norm(
                        feature_weights[
                            neuron_id
                        ]
                    )
                )

                # Match NNSOM's inverted component-plane convention:
                # darker cells represent larger learned weights.
                inverted = tuple(
                    1
                    - np.asarray(
                        color[:3]
                    )
                )

                ax.fill(
                    positions[
                        0,
                        neuron_id,
                    ]
                    + shapex,
                    positions[
                        1,
                        neuron_id,
                    ]
                    + shapey,
                    facecolor=inverted,
                    edgecolor=(
                        0.85,
                        0.85,
                        0.85,
                    ),
                    linewidth=0.25,
                )

            ax.set_title(
                (
                    f"{feature_prefix}_{feature_id}\n"
                    f"min={vmin:.3f}, max={vmax:.3f}"
                ),
                fontsize=8,
            )

        for ax in axes[
            len(page_features):
        ]:
            ax.set_visible(
                False
            )

        page_number = (
            page_start
            // features_per_page
            + 1
        )

        fig.suptitle(
            (
                "SOM Component Planes "
                f"(page {page_number})"
            ),
            fontsize=15,
        )

        fig.tight_layout(
            rect=[
                0,
                0,
                1,
                0.97,
            ]
        )

        output_path = (
            output_dir
            / (
                "component_planes_"
                f"page_{page_number:02d}"
            )
        )

        saved.extend(
            save_figure(
                fig,
                output_path,
                dpi=250,
            )
        )

    return saved


def top_variance_features(
    scaled_train_features: np.ndarray,
    n_features: int,
) -> list[int]:
    """Return indices of the most variable scaled embedding dimensions."""

    variances = np.var(
        scaled_train_features,
        axis=0,
    )

    order = np.argsort(
        variances
    )[::-1]

    return [
        int(i)
        for i in order[
            :n_features
        ]
    ]


def save_class_distribution_maps(
    som,
    data_dict: dict,
    output_dir: Path,
    *,
    num_classes: int = 10,
    include_color: bool = True,
    include_gray: bool = True,
) -> list[Path]:
    """Generate one NNSOM class-distribution map per MNIST digit."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
                        / f"digit_{digit}_color_hist"
                    ),
                    data_dict=data_dict,
                    title=(
                        "MNIST Digit "
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
                        / f"digit_{digit}_gray_hist"
                    ),
                    data_dict=data_dict,
                    title=(
                        "MNIST Digit "
                        f"{digit} Distribution "
                        "(Gray Hist)"
                    ),
                    target_class=digit,
                )
            )

    return saved


def save_feature_hist_maps(
    som,
    data_dict: dict,
    feature_indices: list[int],
    output_dir: Path,
    *,
    include_color: bool = True,
    include_gray: bool = True,
) -> list[Path]:
    """Generate professor-style hist maps for selected embedding dimensions."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = []

    for feature_id in feature_indices:
        if include_color:
            saved.extend(
                save_nnsom_plot(
                    som,
                    "color_hist",
                    (
                        output_dir
                        / (
                            "fc2_"
                            f"{feature_id:02d}_"
                            "color_hist"
                        )
                    ),
                    data_dict=data_dict,
                    title=(
                        "Average LeNet fc2 "
                        f"Feature {feature_id}"
                    ),
                    ind=feature_id,
                )
            )

        if include_gray:
            saved.extend(
                save_nnsom_plot(
                    som,
                    "gray_hist",
                    (
                        output_dir
                        / (
                            "fc2_"
                            f"{feature_id:02d}_"
                            "gray_hist"
                        )
                    ),
                    data_dict=data_dict,
                    title=(
                        "Average LeNet fc2 "
                        f"Feature {feature_id} "
                        "(Gray Hist)"
                    ),
                    ind=feature_id,
                )
            )

    return saved


def compute_neuron_classification_stats(
    clusters,
    labels: np.ndarray,
    preds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-neuron MNIST statistics using NNSOM utilities."""

    class_counts = np.asarray(
        count_classes_in_cluster(
            labels,
            clusters,
        ),
        dtype=np.float64,
    )

    support = np.sum(
        class_counts,
        axis=1,
    ).astype(
        np.float64
    )

    dominant_raw = (
        majority_class_cluster(
            labels,
            clusters,
        )
    )

    dominant_true_class = np.asarray(
        [
            (
                np.nan
                if value is None
                else float(value)
            )
            for value in dominant_raw
        ],
        dtype=np.float64,
    )

    max_class_count = np.max(
        class_counts,
        axis=1,
    )

    purity = np.divide(
        max_class_count,
        support,
        out=np.zeros_like(
            max_class_count,
            dtype=np.float64,
        ),
        where=support > 0,
    )

    error_percent = np.asarray(
        get_perc_misclassified(
            labels,
            preds,
            clusters,
        ),
        dtype=np.float64,
    )

    error_rate = (
        error_percent
        / 100.0
    )

    misclassified_indices = np.asarray(
        get_ind_misclassified(
            labels,
            preds,
        ),
        dtype=np.int64,
    )

    edge_width_raw = (
        get_edge_widths(
            misclassified_indices,
            clusters,
        )
    )

    edge_width = np.asarray(
        [
            (
                0.0
                if value is None
                else float(value)
            )
            for value in edge_width_raw
        ],
        dtype=np.float64,
    )

    # Build per-neuron subsets containing only misclassified samples.
    wrong_clusters = []

    for cluster in clusters:
        cluster_indices = np.asarray(
            cluster,
            dtype=np.int64,
        )

        wrong_clusters.append(
            np.intersect1d(
                cluster_indices,
                misclassified_indices,
            )
        )

    dominant_wrong_raw = (
        majority_class_cluster(
            preds,
            wrong_clusters,
        )
    )

    dominant_wrong_prediction = []

    for neuron_id, value in enumerate(
        dominant_wrong_raw
    ):
        if value is None:
            if np.isnan(
                dominant_true_class[
                    neuron_id
                ]
            ):
                dominant_wrong_prediction.append(
                    np.nan
                )
            else:
                dominant_wrong_prediction.append(
                    dominant_true_class[
                        neuron_id
                    ]
                )
        else:
            dominant_wrong_prediction.append(
                float(value)
            )

    dominant_wrong_prediction = (
        np.asarray(
            dominant_wrong_prediction,
            dtype=np.float64,
        )
    )

    return {
        "support": support,
        "purity": purity,
        "error_rate": error_rate,
        "edge_width": edge_width,
        "dominant_true_class": (
            dominant_true_class
        ),
        "dominant_wrong_prediction": (
            dominant_wrong_prediction
        ),
    }


def save_simple_grid_map(
    som,
    data_dict: dict,
    values: np.ndarray,
    sizes: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> list[Path]:
    """Save an NNSOM simple-grid map."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    sizes = np.asarray(
        sizes,
        dtype=np.float64,
    )

    expected_shape = (
        int(som.numNeurons),
    )

    if values.shape != expected_shape:
        raise ValueError(
            "simple_grid values must have one "
            "value per SOM neuron."
        )

    if sizes.shape != expected_shape:
        raise ValueError(
            "simple_grid sizes must have one "
            "value per SOM neuron."
        )

    if np.max(
        sizes
    ) <= 0:
        raise ValueError(
            "simple_grid requires at least "
            "one non-empty neuron."
        )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    plot_dict = dict(
        data_dict
    )

    # NNSOM plot("simple_grid") expects:
    # column 0 = color value
    # column 1 = hexagon size value
    plot_dict[
        "add_2d_array"
    ] = np.column_stack(
        [
            values,
            sizes,
        ]
    )

    return save_nnsom_plot(
        som,
        "simple_grid",
        output_path,
        data_dict=plot_dict,
        title=title,
        use_add_array=True,
    )


def save_complex_error_map(
    som,
    data_dict: dict,
    stats: dict[str, np.ndarray],
    output_path: Path,
) -> list[Path]:
    """Save a multiclass MNIST complex-hit map."""

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
