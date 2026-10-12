"""Text styling for NNSOM's dense SOM plots (neuron labels on hexagons).

Split from utils/som_plotting.py (2026-10-10). Used by som_plotting.save_nnsom_plot:
NNSOM draws one small label per neuron; these helpers pick a readable size and a
light or dark outline for each label, based on the colour behind it.
"""

from __future__ import annotations

from matplotlib.text import Text

from collections.abc import Sequence

import matplotlib.patheffects as path_effects



def relative_luminance(rgba: Sequence[float]) -> float:
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


def uhi_label_style(
    text_obj: Text,
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


def restyle_dense_som_text(result: object, plot_type: str) -> None:
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
            uhi_label_style(
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
                        relative_luminance(
                            facecolor
                        )
                        < 0.34
                    )

                except Exception:
                    dark_background = False

            uhi_label_style(
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
                        relative_luminance(
                            facecolor
                        )
                        < 0.34
                    )

                except Exception:
                    dark_background = False

            uhi_label_style(
                text_obj,
                font_size=font_size,
                dark_background=dark_background,
            )
