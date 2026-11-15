"""Assign embeddings to the neurons (cells) of the selected SOM.

The same steps as the Phase 2 cluster analysis: load the SOM (its hash is checked first), call
``cluster_data`` on the raw embeddings, and turn the per-neuron lists into one neuron per image.
"""

from __future__ import annotations

import numpy as np

from src.v1_mnist.component.utils.run_settings import (
    AnalysisSettings,
    verify_selected_model,
)
from src.v1_mnist.component.utils.som_io import (
    load_selected_som,
    reconstruct_assignments,
)


def assign_cells(analysis: AnalysisSettings, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Neuron index and distance to the neuron for every embedding of ``features``."""
    verify_selected_model(analysis)
    som = load_selected_som(
        analysis.model_path, analysis.grid_height, analysis.grid_width, analysis.feature_dim
    )
    clusters, distances, _max_distance, sizes = som.cluster_data(features)
    if int(np.asarray(sizes).sum()) != len(features):
        raise RuntimeError("Cluster sizes do not add up to the number of embeddings.")
    return reconstruct_assignments(clusters, distances, len(features))
