# MNIST v1 Source Code (`src/v1_mnist/`)

This directory contains the core source code, pipelines, and tests for the MNIST baseline and NNSOM post-training analysis.

## Structure

- **`component/`**: the code, arranged by project phase and week (`p` = phase; see `component/README.md`)
  - `pipeline/`: one runner per week, plus `run_all.py` (see `component/pipeline/README.md`)
  - `p1_model_training_baseline/`: `data_pipeline/` (Week 1), `cnn_baseline_training/` (Week 2)
  - `p2_som_training_visualization/`: `som_nnsom_training/` (Week 3), `cluster_analysis/` (Week 4), `error_pattern_analysis/` (Week 5), `interactive_report/` (Weeks 3–5)
  - `p3_novelty_detection/`, `p4_guided_retraining/`: week folders for Weeks 6–11 (empty for now)
  - `configs/`: YAML configurations (`base.yaml`, `cnn_baseline.yaml`, `som.yaml`, robustness configs)
  - `utils/`: config loader, logging, run settings, shared SOM I/O, statistics and heatmap helpers
- **`tests/`**: unit and regression tests, in the same phase/week folders (see `tests/README.md`)
  - `p1_model_training_baseline/`: `data_pipeline/` (Week 1), `cnn_baseline_training/` (Week 2)
  - `p2_som_training_visualization/`: `som_nnsom_training/` (Week 3), `cluster_analysis/` (Week 4), `error_pattern_analysis/` (Week 5)
  - `p3_novelty_detection/`, `p4_guided_retraining/`: ready for upcoming tests
- **`docs/`**: documents, in the same phase/week folders (see `docs/README.md`)
  - `p1_model_training_baseline/cnn_baseline_training/`: `cnn_baseline.md` (Week 2)
  - `p2_som_training_visualization/som_nnsom_training/`: SOM grid-size protocol, pre-registered fair test and results (Week 3)
  - `p2_som_training_visualization/error_pattern_analysis/`: error-pattern robustness criteria and results (Week 5)
  - `DEVIATIONS.md`: project-wide log of rule deviations and changes
- **`shellscripts/`**: Convenience shell scripts
  - `run_cnn_baseline.sh`: runs the Week 1 and Week 2 runners
