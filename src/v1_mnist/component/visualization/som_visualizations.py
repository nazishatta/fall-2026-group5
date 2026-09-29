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
import matplotlib.patheffects as path_effects
import numpy as np
from NNSOM.utils import (
    count_classes_in_cluster,
    get_edge_widths,
    get_hexagon_shape,
    get_ind_misclassified,
    get_perc_misclassified,
    majority_class_cluster,
)


def _relative_luminance(rgba: Sequence[float]) -> float:
    """Return WCAG-style relative luminance for an RGBA/RGB color."""

    r, g, b = rgba[:3]

    def _linearize(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92

        return (
            (channel + 0.055)
            / 1.055
        ) ** 2.4

    return (
        0.2126 * _linearize(r)
        + 0.7152 * _linearize(g)
        + 0.0722 * _linearize(b)
    )


def _uhi_label_style(
    text_obj,
    *,
    font_size: float,
    dark_background: bool = False,
) -> None:
    """Apply a UHI-map-inspired editorial label style."""

    text_obj.set_fontfamily(
        "DejaVu Sans"
    )

    text_obj.set_fontsize(
        font_size
    )

    text_obj.set_fontweight(
        "semibold"
    )

    if dark_background:
        # White text with a subtle dark outline on dark fills.
        text_obj.set_color(
            "white"
        )

        text_obj.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=1.15,
                    foreground="#4d4d4d",
                ),
                path_effects.Normal(),
            ]
        )

    else:
        # UHI-map style: dark gray text with a white halo.
        text_obj.set_color(
            "#4d4d4d"
        )

        text_obj.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=1.35,
                    foreground="white",
                ),
                path_effects.Normal(),
            ]
        )


def _restyle_dense_som_text(result, plot_type: str) -> None:
    """Apply UHI-map-inspired typography to dense 15x15 SOM plots.

    Style:
    - clean sans-serif font
    - semibold labels
    - dark-gray text with a white halo on light/mid backgrounds
    - white text with a subtle dark outline on dark backgrounds
    - adaptive font sizes for multi-digit counts
    """

    if not (
        isinstance(result, tuple)
        and len(result) >= 4
    ):
        return

    text_objects = result[3]

    if text_objects is None:
        return

    # ---------------------------------------------------------
    # Numbered SOM topology
    # ---------------------------------------------------------
    if plot_type == "top_num":
        for text_obj in text_objects:
            _uhi_label_style(
                text_obj,
                font_size=6.2,
                dark_background=False,
            )

    # ---------------------------------------------------------
    # Train / validation hit histograms
    # ---------------------------------------------------------
    elif plot_type == "hit_hist":
        patches = (
            result[2]
            if len(result) > 2
            else None
        )

        for neuron_id, text_obj in enumerate(
            text_objects
        ):
            label = (
                text_obj
                .get_text()
                .strip()
            )

            if len(label) >= 4:
                font_size = 5.0
            elif len(label) == 3:
                font_size = 5.6
            else:
                font_size = 6.2

            dark_background = False

            if patches is not None:
                try:
                    patch_group = patches[
                        neuron_id
                    ]

                    # NNSOM may return one patch or a list/tuple of patches.
                    if isinstance(
                        patch_group,
                        (list, tuple),
                    ):
                        patch = patch_group[-1]
                    else:
                        patch = patch_group

                    facecolor = (
                        patch.get_facecolor()
                    )

                    dark_background = (
                        _relative_luminance(
                            facecolor
                        )
                        < 0.34
                    )

                except Exception:
                    dark_background = False

            _uhi_label_style(
                text_obj,
                font_size=font_size,
                dark_background=dark_background,
            )

    # ---------------------------------------------------------
    # Complex error-geography map
    # ---------------------------------------------------------
    elif plot_type == "complex_hist":
        patches = (
            result[2]
            if len(result) > 2
            else None
        )

        for neuron_id, text_obj in enumerate(
            text_objects
        ):
            label = (
                text_obj
                .get_text()
                .strip()
            )

            # Slightly larger than regular hit-hist labels because
            # this figure is intended for close analytical reading.
            if len(label) >= 4:
                font_size = 5.4
            elif len(label) == 3:
                font_size = 6.0
            else:
                font_size = 6.6

            dark_background = False

            if patches is not None:
                try:
                    patch_group = patches[
                        neuron_id
                    ]

                    if isinstance(
                        patch_group,
                        (list, tuple),
                    ):
                        patch = patch_group[-1]
                    else:
                        patch = patch_group

                    facecolor = (
                        patch.get_facecolor()
                    )

                    dark_background = (
                        _relative_luminance(
                            facecolor
                        )
                        < 0.34
                    )

                except Exception:
                    dark_background = False

            _uhi_label_style(
                text_obj,
                font_size=font_size,
                dark_background=dark_background,
            )

RECOGNIZED_CATEGORIES = {
    "core",
    "class_maps",
    "feature_maps",
    "component_planes",
    "analysis_maps",
    "extended",
    "convergence",
}


def _route_dual_format_paths(output_path: Path) -> tuple[Path, Path]:
    """Route an output path to twin SVG and PDF paths under svg/ and pdf/ folders."""
    path = Path(output_path)
    parts = list(path.parts)
    dir_parts = parts[:-1]

    if "svg" in dir_parts:
        idx = dir_parts.index("svg")
        pdf_parts = list(parts)
        pdf_parts[idx] = "pdf"
        svg_path = Path(*parts).with_suffix(".svg")
        pdf_path = Path(*pdf_parts).with_suffix(".pdf")
        return svg_path, pdf_path

    if "pdf" in dir_parts:
        idx = dir_parts.index("pdf")
        svg_parts = list(parts)
        svg_parts[idx] = "svg"
        svg_path = Path(*svg_parts).with_suffix(".svg")
        pdf_path = Path(*parts).with_suffix(".pdf")
        return svg_path, pdf_path

    parent_name = path.parent.name
    if parent_name in RECOGNIZED_CATEGORIES:
        root = path.parent.parent
        stem = path.stem
        svg_path = root / "svg" / parent_name / f"{stem}.svg"
        pdf_path = root / "pdf" / parent_name / f"{stem}.pdf"
        return svg_path, pdf_path

    stem = path.stem
    svg_path = path.parent / "svg" / f"{stem}.svg"
    pdf_path = path.parent / "pdf" / f"{stem}.pdf"
    return svg_path, pdf_path


def _sanitize_figure_for_export(fig) -> None:
    """Sanitize artists in a figure for strict vector (SVG/PDF) backends.

    - Clamps RGBA values to [0.0, 1.0] to prevent SVG hex-conversion errors.
    - Replaces NaN/Inf in line widths, colors, or coordinates to prevent PDF backend errors.
    """
    for ax in fig.axes:
        for patch in ax.patches:
            try:
                fc = patch.get_facecolor()
                if fc is not None:
                    fc_arr = np.asarray(fc, dtype=float)
                    if not np.all(np.isfinite(fc_arr)) or np.any(fc_arr < 0.0) or np.any(fc_arr > 1.0):
                        patch.set_facecolor(np.clip(np.nan_to_num(fc_arr, nan=0.0), 0.0, 1.0))
            except Exception:
                pass

            try:
                ec = patch.get_edgecolor()
                if ec is not None:
                    ec_arr = np.asarray(ec, dtype=float)
                    if not np.all(np.isfinite(ec_arr)) or np.any(ec_arr < 0.0) or np.any(ec_arr > 1.0):
                        patch.set_edgecolor(np.clip(np.nan_to_num(ec_arr, nan=0.0), 0.0, 1.0))
            except Exception:
                pass

            try:
                lw = patch.get_linewidth()
                if lw is not None and not np.isfinite(lw):
                    patch.set_linewidth(0.0)
            except Exception:
                pass

        for coll in ax.collections:
            try:
                fcs = coll.get_facecolors()
                if fcs is not None and len(fcs) > 0:
                    fcs_arr = np.asarray(fcs, dtype=float)
                    if not np.all(np.isfinite(fcs_arr)) or np.any(fcs_arr < 0.0) or np.any(fcs_arr > 1.0):
                        coll.set_facecolors(np.clip(np.nan_to_num(fcs_arr, nan=0.0), 0.0, 1.0))
            except Exception:
                pass

            try:
                ecs = coll.get_edgecolors()
                if ecs is not None and len(ecs) > 0:
                    ecs_arr = np.asarray(ecs, dtype=float)
                    if not np.all(np.isfinite(ecs_arr)) or np.any(ecs_arr < 0.0) or np.any(ecs_arr > 1.0):
                        coll.set_edgecolors(np.clip(np.nan_to_num(ecs_arr, nan=0.0), 0.0, 1.0))
            except Exception:
                pass

            try:
                lws = coll.get_linewidths()
                if lws is not None and len(lws) > 0:
                    lws_arr = np.asarray(lws, dtype=float)
                    if not np.all(np.isfinite(lws_arr)):
                        coll.set_linewidths(np.nan_to_num(lws_arr, nan=0.0))
            except Exception:
                pass

        for line in ax.lines:
            try:
                color = line.get_color()
                if isinstance(color, (list, tuple, np.ndarray)):
                    c_arr = np.asarray(color, dtype=float)
                    if not np.all(np.isfinite(c_arr)) or np.any(c_arr < 0.0) or np.any(c_arr > 1.0):
                        line.set_color(np.clip(np.nan_to_num(c_arr, nan=0.0), 0.0, 1.0))
            except Exception:
                pass
            try:
                lw = line.get_linewidth()
                if not np.isfinite(lw):
                    line.set_linewidth(1.0)
            except Exception:
                pass


def save_figure(
    fig,
    output_path: Path,
    *,
    dpi: int = 300,
    also_pdf: bool = True,
    formats: tuple[str, ...] = ("svg", "pdf"),
) -> list[Path]:
    """Save a Matplotlib figure as vector SVG and PDF under dedicated directories and close it."""

    _sanitize_figure_for_export(fig)
    svg_path, pdf_path = _route_dual_format_paths(output_path)

    saved: list[Path] = []

    if "svg" in formats:
        svg_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            svg_path,
            bbox_inches="tight",
        )
        saved.append(svg_path)

    if "pdf" in formats or also_pdf:
        pdf_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )
        saved.append(pdf_path)

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

    _restyle_dense_som_text(
        result,
        plot_type,
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
        title_size = (
            12
            if plot_type == "complex_hist"
            else 13
            if plot_type in {
                "top_num",
                "hit_hist",
            }
            else 14
        )

        fig.suptitle(
            title,
            fontsize=title_size,
            fontfamily="DejaVu Sans",
            fontweight="semibold",
            color="#222222",
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
            fontfamily="DejaVu Sans",
            fontweight="medium",
            color="#4d4d4d",
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
