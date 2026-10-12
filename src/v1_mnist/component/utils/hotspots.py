"""The project's error-hotspot rule (Week 5, Error Pattern Analysis).

One rule everywhere (docs/p2_som_training_visualization/error_pattern_analysis/HOTSPOT_RULE.md):
rank the occupied neurons by the 95% Wilson lower bound of their error rate
(ties: more errors, then more images, then lower neuron index) and take the top
k = max(5, round(5% of neurons)): 5 on 10x10, 11 on 15x15, 20 on 20x20.

This is the rule fixed in advance in ERROR_PATTERN_ROBUSTNESS_CRITERIA.md
(section 3, step 2). The robustness test (error_pattern_robustness.py) applies it
to half-splits of the validation images; Week 5 (error_patterns.py) applies it to
the full validation split to list the hotspots. The code was moved here unchanged
from error_pattern_robustness.py.
"""

from __future__ import annotations

import numpy as np

from src.v1_mnist.component.utils.stats import wilson_lower_bound_array

# The rule's two numbers come from the SOM configs (analysis.hotspot_fraction: 0.05,
# analysis.min_hotspots: 5), where they are marked as fixed in advance.


def n_hotspots(num_neurons: int, hotspot_fraction: float, min_hotspots: int) -> int:
    """Number of hotspots k = max(min_hotspots, round(hotspot_fraction x neurons))."""
    return max(min_hotspots, int(round(hotspot_fraction * num_neurons)))


def top_hotspots(bmu: np.ndarray, err: np.ndarray, num_neurons: int, k: int) -> np.ndarray:
    """Top-k occupied neurons: Wilson bound desc, then errors desc, support desc, index asc."""
    support = np.bincount(bmu, minlength=num_neurons)
    errors = np.bincount(bmu, weights=err, minlength=num_neurons)
    lb = wilson_lower_bound_array(errors, support)
    order = np.lexsort((np.arange(num_neurons), -support, -errors, -lb))
    order = order[support[order] > 0]
    return order[:k]
