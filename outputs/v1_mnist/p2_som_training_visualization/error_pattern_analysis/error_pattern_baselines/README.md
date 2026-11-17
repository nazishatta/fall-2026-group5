# Error-pattern baselines (`outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/error_pattern_baselines/`)

Is the SOM better at finding the CNN's mistakes than simpler methods? The SOM's error cells are compared with k-means cells (10 seeds), kNN label impurity and max-softmax confidence on the same validation images. Plan: `src/v1_mnist/docs/p2_som_training_visualization/error_pattern_analysis/baselines/PLAN.md`. Runner: `pipeline/p2_som_training_visualization/run_error_pattern_baselines.py`.

Run folder `error_pattern_baselines_som_15x15/`:

| Path | What is inside |
|---|---|
| `results.json` | All results, with intervals |
| `summary.csv` | One row per method: the table used in the report |
| `work/` (+ `work.dvc`) | The result of each method and seed (DVC) |
| `logs/` (+ `logs.dvc`) | Run logs (DVC) |

Main result: max-softmax finds mistakes better than the SOM (AUROC 0.981 against 0.685).
