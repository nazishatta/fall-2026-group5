# Error pattern analysis results (`outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/`)

Results of Phase 2, Error Pattern Analysis (code: `src/v1_mnist/component/p2_som_training_visualization/error_pattern_analysis/`).

| Path | What is inside |
|---|---|
| `som_15x15_nb11_ep250_s42_v2/` | The headline map: `error_summary.json`; `tables/` (error rate per neuron, hotspots, confusion pairs, misclassified images); `representative_samples/` (images in the hotspots); `metadata/` (SHA-256 fingerprints) |
| [`robustness/`](robustness/README.md) | The same analysis on the 10x10, 15x15 and 20x20 maps, and the verdict |
| [`error_pattern_baselines/`](error_pattern_baselines/README.md) | SOM error cells against k-means cells, kNN impurity and max-softmax |
| [`error_detector_budget/`](error_detector_budget/README.md) | The same detectors compared at the same share of flagged images |
| `figures/` (+ `figures.dvc`) | Error-rate and error maps, error heatmaps (PDF, DVC) |
| `logs/` | Logs of the figure scripts |
