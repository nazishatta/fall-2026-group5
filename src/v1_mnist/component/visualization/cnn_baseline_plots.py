from __future__ import annotations

"""CNN Baseline Plotting Helper.

Re-exports vector SVG plotting routines from plots.py for backward compatibility.
"""

from src.v1_mnist.component.visualization.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_training_curves,
    plot_sample_images,
)

__all__ = [
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_training_curves",
    "plot_sample_images",
]
