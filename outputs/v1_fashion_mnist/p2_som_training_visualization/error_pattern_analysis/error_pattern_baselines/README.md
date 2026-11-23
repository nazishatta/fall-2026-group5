# Error-pattern baselines (Fashion-MNIST)

Is the SOM better at finding the CNN's mistakes than simpler methods? The SOM's error cells are compared with k-means cells (10 seeds), nearest-neighbour label impurity and max-softmax confidence on the same 10,500 validation images (876 mistakes). Plan: `src/v1_mnist/docs/p2_som_training_visualization/error_pattern_analysis/baselines/PLAN.md`, followed unchanged. Runner: `src/v1_mnist/component/pipeline/p2_som_training_visualization/run_error_pattern_baselines.py`. Config: `src/v1_fashion_mnist/component/configs/error_pattern_baselines.yaml`. Exploratory, as on MNIST.

Run folder `error_pattern_baselines_som_15x15/`:

| Path | What is inside |
|---|---|
| `results.json` | All results, with intervals |
| `summary.csv` | One row per method: AUROC, average precision, hotspot ratio and the share of images and mistakes in the hotspots |
| `error_pattern_baselines.tex` (+ `.dvc`) | The same table as LaTeX (DVC) |
| `work/` (+ `work.dvc`) | The result of each method and seed (DVC) |
| `logs/` (+ `logs.dvc`) | Run logs (DVC) |

Main result: max-softmax finds the mistakes best (AUROC 0.906), then nearest-neighbour impurity (0.876); SOM cells (0.811) and k-means cells (0.812) are equal. Recorded in the Phase 2 protocol, section 5.
