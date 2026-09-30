# MNIST v1 Source Code (`src/v1_mnist/`)

This directory contains the core source code, pipelines, and tests for the MNIST baseline and NNSOM post-training analysis.

## Structure

- **`component/`**: Modular application logic
  - `data/`: Dataset loading and stratified splitting for MNIST
  - `models/`: CNN architectures (LeNet-5)
  - `features/`: Feature extraction hook (`fc2` layer, 84-D)
  - `evaluation/`: Accuracy, Macro-F1, confusion matrix computation
  - `visualization/`: Training curves and confusion heatmap plotting
  - `som/`: NNSOM trainer, dataset adapters, and memory-safe initialization
  - `analysis/`: SOM grid selection protocol and BMU/quantization analysis
  - `pipeline/`: Sequentially executable CLI pipelines (`01_prepare_data.py` through `05_visualize_som.py`)
  - `configs/`: YAML configurations (`base.yaml`, `cnn_baseline.yaml`, `som.yaml`)
  - `utils/`: Configuration loader and output path resolver
- **`tests/`**: Unit and regression tests
  - `cnn_baseline/`: Dataset split, LeNet-5 forward pass, and feature extractor tests
  - `som/`: BMU analysis, memory-safe init, and scientific run isolation tests
- **`docs/`**: Technical documentation and protocols
  - `cnn_baseline.md`: Technical documentation for LeNet-5 baseline
  - `som/SOM_GRID_SELECTION_PROTOCOL.md`: Protocol for SOM grid selection
- **`shellscripts/`**: Convenience shell scripts
  - `run_cnn_baseline.sh`: Automated pipeline execution for CNN baseline
