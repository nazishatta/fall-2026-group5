# Interactive SOM Visualizations — Reproduction Guide

This directory contains interactive Plotly visualizations for the selected MNIST SOM run `som_15x15_seed42_final`.

The visualizations use already-versioned development artifacts. They do not retrain the baseline neural network, retrain the SOM, recompute BMUs, or evaluate the final test split.

## Current Interactive Artifacts

### 1. Quantization Error Convergence

Generated output: `som_qe_convergence.html`

Generator:

`src/v1_mnist/component/visualization/generate_som_qe_interactive.py`

Purpose:

- Visualize the recorded SOM training quantization-error history.
- Show final training and validation quantization error as reference values.
- Inspect training behavior without recomputing the SOM.

This is a training diagnostic. It is not direct evidence for representation structure, novelty detection, or OOD performance.

### 2. Neighborhood Radius Schedule

Generated output: `som_neighborhood_radius.html`

Generator:

`src/v1_mnist/component/visualization/generate_som_radius_interactive.py`

Purpose:

- Visualize the recorded neighborhood-radius schedule.
- Support epoch-level inspection of radius and quantization error.
- Document the optimization schedule of the selected SOM run.

This is also a training diagnostic rather than direct RQ1, RQ2, or OOD evidence.

### 3. Train vs Validation SOM Quality Summary

Generated output: `som_train_validation_quality.html`

Generator:

`src/v1_mnist/component/visualization/generate_som_quality_interactive.py`

Purpose:

- Compare train and validation quantization error.
- Compare first-order topological error.
- Compare mean hits per occupied neuron.
- Compare maximum hits per neuron.
- Expose neuron occupancy through interactive hover information.

The comparison is descriptive for this selected run and is not statistical evidence across multiple seeds.

## Data Integrity

The workflow uses selected SOM development artifacts under:

`outputs/v1_mnist/som/`

The interactive artifact loader requires `test_evaluated == false`.

The held-out final test split is therefore not used to generate these development visualizations.

These development visualizations must not be used to select a final test result, novelty/OOD threshold, retraining strategy, augmentation strategy, best seed, or final model checkpoint.

## Reproduction

From the repository root, install the project requirements:

    python -m pip install -r requirements.txt

Regenerate the interactive outputs:

    python -m src.v1_mnist.component.visualization.generate_som_qe_interactive
    python -m src.v1_mnist.component.visualization.generate_som_radius_interactive
    python -m src.v1_mnist.component.visualization.generate_som_quality_interactive

The generated HTML files are written to:

`demo/fig/v1_mnist/som_interactive_plots/`

## Validation Tests

Run:

    python -m pytest -q src/v1_mnist/tests/test_som_interactive_visualizations.py

The tests currently verify:

- Selected run metadata.
- 15 × 15 SOM grid.
- 225-neuron topology.
- Expected 251 recorded QE-history rows.
- QE figure construction.
- Neighborhood-radius figure construction.
- Train-vs-validation quality figure construction.
- Development-only figure labeling.
- Failure behavior for a missing metrics artifact.

## Scientific Scope and Current Limitation

The current interactive artifacts document SOM training behavior and development-set quality.

They do not yet provide neuron-level interactive RQ1/RQ2 analysis because the exact frozen SOM model artifact required for BMU recomputation is DVC-managed and is currently unavailable locally due to S3 access permissions.

Once the frozen model artifact becomes available, neuron-level interactive analysis should be generated from the actual saved model and analysis tables rather than reconstructed from static PNG figures.
