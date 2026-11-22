# `error_pattern_analysis` results (Fashion-MNIST)

Where the CNN's mistakes sit on the map (Phase 2, RQ2). Code: `src/v1_mnist/component/p2_som_training_visualization/error_pattern_analysis/` (shared with MNIST). Protocol: `src/v1_fashion_mnist/docs/p2_som_training_visualization/SOM_ANALYSIS_PROTOCOL.md`.

| Path | What is inside |
|---|---|
| `som_15x15_nb11_ep250_s42/` | The headline map: `error_summary.json`; `tables/` (error rate per neuron, hotspots, confusion pairs, misclassified images); `figures/` (+ `figures.dvc`: error count, error rate, hotspot map, mean confidence and Wilson lower bound per neuron, PDF); `representative_samples/` (+ `.dvc`: the images in each hotspot, PDF); `metadata/` (SHA-256 fingerprints) |
| `robustness/` | The held-out hotspot check on the 10x10, 15x15 and 20x20 maps, and the verdict |
| [`error_pattern_baselines/`](error_pattern_baselines/README.md) | SOM error cells against k-means cells, nearest-neighbour impurity and max-softmax |
| [`error_detector_budget/`](error_detector_budget/README.md) | The same detectors compared at the same share of flagged images |
| `figures/` (+ `figures.dvc`) | Error-rate and complex error maps (PDF) |

Large files and figures are kept with DVC; the `.json` and `.csv` results are in git.
