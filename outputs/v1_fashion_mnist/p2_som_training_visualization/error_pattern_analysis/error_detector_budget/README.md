# Same-budget comparison of the error detectors (Fashion-MNIST)

Each detector may flag the same share of validation images as the SOM's hotspot cells (and, separately, as the k-means hotspot cells); how many of the mistakes does each catch? Ties at the cut-off are counted by their expected value. Runner: `src/v1_mnist/component/pipeline/p2_som_training_visualization/run_error_detector_budget.py`. Config: `src/v1_fashion_mnist/component/configs/error_detector_budget.yaml`. Exploratory, as on MNIST.

Run folder `error_detector_budget_som_15x15/`:

| Path | What is inside |
|---|---|
| `results.json` | Settings, inputs and all results |
| `summary.csv` | One row per budget and method: share of images, share of mistakes caught with its bootstrap interval, and the ratio |
| `logs/` (+ `logs.dvc`) | Run log (DVC) |

Main result: the SOM hotspot cells flag 3.5% of the images and catch 13.7% of the mistakes; at the same share, nearest-neighbour impurity catches 21.1% and max-softmax 21.9%.
