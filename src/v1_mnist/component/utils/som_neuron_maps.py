"""Per-neuron class and error statistics, and the simple grid map used by the
Week 4 and Week 5 figure scripts (cluster_figures.py, error_figures.py).

Split from utils/som_plotting.py (2026-10-10).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.v1_mnist.component.utils.som_types import SOMModel

from pathlib import Path

import numpy as np
from NNSOM.utils import (
    count_classes_in_cluster,
    get_edge_widths,
    get_ind_misclassified,
    get_perc_misclassified,
    majority_class_cluster,
)


from src.v1_mnist.component.utils.som_plotting import save_nnsom_plot


def compute_neuron_classification_stats(
    clusters: Sequence[Sequence[int]],
    labels: np.ndarray,
    preds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-neuron class and error statistics using NNSOM utilities."""

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
    som: SOMModel,
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
