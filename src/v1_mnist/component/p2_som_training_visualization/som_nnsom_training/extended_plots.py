"""Week 3 extended NNSOM plots (dense hist / box / violin / scatter / weight plots).

Split from visualize_som.py (2026-10-10). Drawn only with visualize_som.py
--extended (or the config switch): they are hard to read on large maps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.v1_mnist.component.utils.som_types import SOMModel

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.v1_mnist.component.utils.som_plotting import save_nnsom_plot


def attempt_extended_plots(
    som: SOMModel,
    data_dict: dict,
    feature_indices: list[int],
    output_dir: Path,
    *,
    layer: str,
    feature_dim: int,
) -> tuple[list[str], dict[str, str]]:
    """Attempt dense NNSOM plots, recording any failures instead of aborting.

    The class-composition plots (class pie and stem) are Week 4 and are drawn
    by cluster_analysis/cluster_figures.py.
    """

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
            "scatter",
            {
                "ind": [
                    f0,
                    f1,
                ],
                "reg_line": False,
            },
            (
                f"scatter_{layer}_"
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
                    f"hist_{layer}_"
                    f"{feature_id:02d}"
                ),
                data_dict=extended_data_dict,
                ind=feature_id,
                title=(
                    "Cluster Histogram — "
                    f"{layer} Feature {feature_id}"
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
                    f"hist_{layer}_"
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
            / f"box_top_{layer}_features",
            data_dict=extended_data_dict,
            ind=[
                int(i)
                for i in feature_indices
            ],
            title=(
                "Box Plot — "
                f"Top-Variance {layer} Features"
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
            f"box_top_{layer}_features"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    try:
        paths = save_nnsom_plot(
            som,
            "violin",
            output_dir
            / f"violin_top_{layer}_features",
            data_dict=extended_data_dict,
            ind=[
                int(i)
                for i in feature_indices
            ],
            title=(
                "Violin Plot — "
                f"Top-Variance {layer} Features"
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
            f"violin_top_{layer}_features"
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
                f"({feature_dim} {layer} dimensions)"
            ),
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
                f"(first two {layer} dimensions)"
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
