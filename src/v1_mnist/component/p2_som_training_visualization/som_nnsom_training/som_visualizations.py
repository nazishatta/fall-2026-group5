"""Week 3 SOM drawing helpers: component planes and feature maps.

Paginated component-plane views keep every embedding dimension readable on
the selected SOM. The layer and model names come from the config. Shared helpers (saving figures, NNSOM plots)
are in utils/som_plotting.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.v1_mnist.component.utils.som_types import SOMModel

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from NNSOM.utils import (
    get_hexagon_shape,
)

from src.v1_mnist.component.utils.som_plotting import save_figure, save_nnsom_plot


def save_native_component_planes(
    som: SOMModel,
    data_dict: dict,
    output_path: Path,
    *,
    feature_prefix: str,
    model_label: str,
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
            f"{num_features} {model_label} {feature_prefix} Features"
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
    )


def save_component_plane_pages(
    som: SOMModel,
    output_dir: Path,
    *,
    feature_indices: Sequence[int] | None=None,
    features_per_page: int = 12,
    feature_prefix: str,
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
            )
        )

    return saved


def save_feature_hist_maps(
    som: SOMModel,
    data_dict: dict,
    feature_indices: list[int],
    output_dir: Path,
    *,
    include_color: bool = True,
    include_gray: bool = True,
    feature_prefix: str,
    model_label: str,
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
                            f"{feature_prefix}_"
                            f"{feature_id:02d}_"
                            "color_hist"
                        )
                    ),
                    data_dict=data_dict,
                    title=(
                        f"Average {model_label} {feature_prefix} "
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
                            f"{feature_prefix}_"
                            f"{feature_id:02d}_"
                            "gray_hist"
                        )
                    ),
                    data_dict=data_dict,
                    title=(
                        f"Average {model_label} {feature_prefix} "
                        f"Feature {feature_id} "
                        "(Gray Hist)"
                    ),
                    ind=feature_id,
                )
            )

    return saved
