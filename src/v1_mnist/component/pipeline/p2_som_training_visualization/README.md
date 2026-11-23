# Phase 2 runners (`src/v1_mnist/component/pipeline/p2_som_training_visualization/`)

One runner per Phase 2 step, named after the step folder it runs (`run_<step>.py` runs `component/p2_som_training_visualization/<step>/`). Run from the code root (the folder that holds `src/` and `outputs/`). Each runner reads its settings from `component/configs/`, checks that its run ID is new, and never overwrites a finished run. See `component/pipeline/README.md` for which runners `run_all.py` calls and which are run on their own.

## Files

| File | What it is |
|---|---|
| `run_cluster_analysis.py` ▶ | Cluster analysis of the trained SOM. |
| `run_error_detector_budget.py` ▶ | Same-budget comparison of the error detectors (exploratory): kNN impurity and max-softmax flag the same share of validation images as the SOM's and k-means' hotspot cells; how many of the CNN's mistakes do they catch? |
| `run_error_pattern_analysis.py` ▶ | Error pattern analysis and its robustness check. |
| `run_error_pattern_baselines.py` ▶ | Error pattern baselines: is the SOM's error-geography result special, or do simple alternatives find it too? |
| `run_feature_layer_ablation.py` ▶ | Feature-layer ablation (Phase 2): SOMs on fc1 and fc2 of the Phase 1 LeNet-5, compared on A1 to A3. |
| `run_grid_comparison.py` ▶ | Pre-registered grid comparison (Phase 2): do the 10x10, 15x15 and 20x20 maps differ in how well their cells predict held-out CNN mistakes? |
| `run_interactive_report.py` ▶ | Interactive HTML report for the trained SOM. |
| `run_som_nnsom_training.py` ▶ | SOM training and visualization with NNSOM. |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
