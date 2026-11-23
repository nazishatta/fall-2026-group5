# `som_nnsom_training` results (Fashion-MNIST)

Phase 2, SOM training. The headline map is `som_15x15_nb11_ep250_s42`: 15x15 grid, starting neighbourhood 11, 250 epochs, seed 42, `MinMaxScaler(-1, 1)` fitted on the training embeddings (config: `src/v1_fashion_mnist/component/configs/som.yaml`). The 10x10 and 20x20 maps are for the grid table and the error-pattern robustness check. Trained with the project's batch SOM (`train_som_with_history`: NNSOM initialisation, winners recomputed every epoch; CPU/NumPy).

| Path | What is inside | Where it is kept |
|---|---|---|
| `som_models/` | The three trained maps | DVC (`som_models.dvc`) |
| `logs/<run>_metrics.json` | Quality measures on training and validation (quantization and topographic error, occupancy) | git |
| `logs/<run>_qe_history.json`, `tables/<run>_qe_history.csv` | Quantization error after every epoch | git |
| `logs/run_manifest.csv` | The maps' SHA-256 fingerprints | git |
| `tables/som_grid_ablation.csv`, `.json` | The 10x10, 15x15 and 20x20 maps side by side | git |
| `figures/<run>/` | Core figures (topology, hit histograms, U-matrix), component planes, feature maps, extended figures (PDF) | DVC (`figures.dvc`) |
| `convergence_plots/` | Interactive convergence plots (HTML) | DVC (`convergence_plots.dvc`) |

Main result (15x15): quantization error 0.915 on training and 0.899 on validation; every neuron receives training images and 98.7% receive validation images. Cluster and error maps are in `../cluster_analysis/` and `../error_pattern_analysis/`.
