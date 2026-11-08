"""Hard example mining: decide which training images are hard, before any retraining.

Training images and Phase 2 records only (no CNN is trained, the test split is never loaded).
Writes the per-image difficulty table, the pool tables, and results.json with the thresholds and
the SHA-256 of every input.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.difficulty import (
    difficulty_flags,
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.inputs import (
    load_train_inputs,
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.pool_outputs import (
    write_difficulty_table,
    write_pool_tables,
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.pool_selection import (
    msp_pool,
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.settings import (
    PoolSettings,
)
from src.v1_mnist.component.utils.run_provenance import ensure_not_done, write_json
from src.v1_mnist.component.utils.run_settings import AnalysisSettings

LOGGER = logging.getLogger("v1_mnist.hard_example_mining")


def build_pools(settings: PoolSettings, analysis: AnalysisSettings, repo_root: Path) -> dict[str, Any]:
    """Run the step once; stops if this run ID already has results."""
    ensure_not_done(settings.results_path)
    data = load_train_inputs(settings, analysis, repo_root)
    LOGGER.info("Loaded %d training images; %d hotspot neurons.", len(data.label), len(data.hotspot_neurons))

    flags = difficulty_flags(
        data.bmu, data.bmu_distance, data.purity_by_neuron, data.hotspot_neurons,
        settings.purity_cutoff, settings.qe_percentile,
    )
    som_size = int(flags.in_som_pool.sum())
    msp = msp_pool(data.confidence, data.sample_id, som_size)
    LOGGER.info("SOM pool: %d images. Distance cut-off: %.4f.", som_size, flags.qe_threshold)

    write_difficulty_table(
        settings.output_dir / "difficulty" / "train_difficulty.csv",
        data, data.purity_by_neuron[data.bmu], flags, msp,
    )
    write_pool_tables(settings.output_dir, data, flags, msp, analysis.num_classes)

    results: dict[str, Any] = {
        "run_id": settings.run_id,
        "selected_som": analysis.selected_model,
        "split": "train",
        "training_images": len(data.label),
        "training_errors": int((~data.correct).sum()),
        "hotspot_neurons": [int(n) for n in data.hotspot_neurons],
        "purity_cutoff": settings.purity_cutoff,
        "qe_percentile": settings.qe_percentile,
        "qe_threshold": flags.qe_threshold,
        "pool_sizes": {
            "in_hotspot": int(flags.in_hotspot.sum()),
            "low_purity": int(flags.low_purity.sum()),
            "qe_flag": int(flags.qe_flag.sum()),
            "som_pool": som_size,
            "msp_pool": int(msp.sum()),
        },
        "sha256": data.sha256,
    }
    write_json(settings.results_path, results)
    LOGGER.info("Wrote %s", settings.results_path)
    return results
