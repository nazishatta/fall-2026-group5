"""Shared SOM plotting helpers for every week.

Used by the Week 3 (som_nnsom_training/visualize_som.py), Week 4
(cluster_analysis/cluster_figures.py) and Week 5
(error_pattern_analysis/error_figures.py) figure scripts, and by the
interactive report. Week-specific drawing functions live in each week's folder.
Label styling: som_text_style.py. Per-neuron class/error statistics and the simple
grid map: som_neuron_maps.py (both split from this file on 2026-10-10).

The functions intentionally use NNSOM's own plotting methods when they
are informative for the dataset.

Important NNSOM behavior:
``hit_hist``, ``gray_hist`` and ``color_hist`` cache ``som.outputs``.
Before each such plot we force a fresh simulation so train and validation
plots cannot accidentally reuse assignments from a previous split.
"""

from __future__ import annotations

from matplotlib.figure import Figure
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.v1_mnist.component.utils.som_types import SOMModel

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.v1_mnist.component.utils.figures import save_pdf

from src.v1_mnist.component.utils.som_text_style import restyle_dense_som_text


def _sanitize_figure_for_export(fig: Figure) -> None:
    """Sanitize artists in a figure for the PDF backend.

    - Clamps RGBA values to [0.0, 1.0] so invalid colours cannot break the export.
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
    fig: Figure,
    output_path: Path,
    *,
    rasterize_dense: bool = False,
) -> list[Path]:
    """Save a Matplotlib figure as one PDF (pdf/ folder, embedded fonts) and close it.

    rasterize_dense: rasterize the dense data layer (axes and text stay vector).
    """

    _sanitize_figure_for_export(fig)
    return [save_pdf(fig, output_path, rasterize_dense=rasterize_dense)]


def build_data_dict(
    som: SOMModel,
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
    som: SOMModel,
) -> None:
    """Prevent NNSOM hit-family plots from reusing stale cached outputs."""

    som.sim_flag = True
    som.outputs = None


def save_nnsom_plot(
    som: SOMModel,
    plot_type: str,
    output_path: Path,
    *,
    data_dict: dict | None = None,
    title: str | None = None,
    **kwargs: Any,
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

    restyle_dense_som_text(
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

    # Saved fully vector. Rasterizing the data layer was tried on 2026-10-10: it broke
    # the layout of NNSOM's violin, scatter and box plots (one small inset chart per
    # neuron) and made the component planes about 10x larger. Every SOM figure PDF is
    # under 2.2 MB without it.
    return save_figure(
        fig,
        output_path,
    )


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
