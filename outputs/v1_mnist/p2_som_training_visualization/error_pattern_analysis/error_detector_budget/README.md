# Same-budget comparison of the error detectors (`outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/error_detector_budget/`)

Each detector may flag the same share of validation images as the SOM's hotspot cells (and, separately, as the k-means hotspot cells); how many of the mistakes does each catch? Ties at the cut-off are counted by their expected value. Runner: `pipeline/p2_som_training_visualization/run_error_detector_budget.py`. Config: `configs/error_detector_budget.yaml`. This analysis is exploratory (see `docs/DEVIATIONS.md`).

Run folder `error_detector_budget_som_15x15/`:

| File | What it is |
|---|---|
| `results.json` | Settings, inputs and all results |
| `summary.csv` | One row per budget and method: share of images, share of mistakes caught with its bootstrap interval, and the ratio |

Main result: the SOM hotspot cells flag 1.6% of the images and catch 18.1% of the mistakes; at the same share, kNN impurity catches 48.7% and max-softmax 53.3%.
