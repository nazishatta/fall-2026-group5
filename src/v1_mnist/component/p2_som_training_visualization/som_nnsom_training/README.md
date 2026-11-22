# SOM training with NNSOM (`src/v1_mnist/component/p2_som_training_visualization/som_nnsom_training/`)

Proposal step: SOM Setup with NNSOM. Trains self-organizing maps (SOMs) with the NNSOM library on the CNN's fc2 embeddings (standardised with a scaler fitted on the training images), records the QE after every epoch, measures map quality (QE and topographic error) and draws the SOM figures.

The headline map is `som_15x15_nb11_ep250_s42_v2` (15x15 grid, starting neighbourhood 11, 250 epochs, seed 42). The 10x10 and 20x20 maps are used for robustness checks. `grid_ablation.py` compares the three sizes; the `optuna_*.py` files tune the starting neighbourhood and the number of epochs with Optuna for all three sizes (protocol: `docs/p2_som_training_visualization/som_nnsom_training/OPTUNA_TUNING_PROTOCOL.md`).

Runner: `pipeline/p2_som_training_visualization/run_som_nnsom_training.py`. Results: `outputs/v1_mnist/p2_som_training_visualization/som_nnsom_training/` and `som_optuna_tuning/`.

## Folders

| Folder | What is inside |
|---|---|
| [`convergence_plots/`](convergence_plots/README.md) | Three interactive HTML charts of how a SOM trained: the quantization error (QE) per epoch, the training and validation quality, and the shrinking neighbourhood radius. |
| [`layer_ablation/`](layer_ablation/README.md) | SOMs on two layers of the frozen Phase 1 LeNet-5, `fc1` (120 numbers) and `fc2` (84 numbers, the layer every other step uses), with the same Phase 2 recipe and 5 SOM seeds each. |
| [`grid_comparison/`](grid_comparison/README.md) | The pre-registered comparison of the 10x10, 15x15 and 20x20 maps on predicting held-out CNN mistakes (10 bootstrap maps per grid, 200 repeats). |

## Files

| File | What it is |
|---|---|
| `extended_plots.py` | Extended NNSOM plots (dense hist / box / violin / scatter / weight plots). |
| `grid_ablation.py` ▶ | Descriptive 10x10 / 15x15 / 20x20 SOM grid ablation table. |
| `memory_safe_init.py` | Makes NNSOM's SVD-based weight start use an economy SVD, so it needs much less memory. |
| `optuna_common.py` | Shared pieces of the Optuna SOM tuning: config, frozen inputs and one scored SOM run. |
| `optuna_confirmation.py` ▶ | Stage 2 of the Optuna SOM tuning: check the chosen setting on new SOM seeds. |
| `optuna_grid_tuning.py` ▶ | Stage 1 of the Optuna SOM tuning: the search, one grid per run (validation features only). |
| `optuna_selection.py` | Selection and comparison rules for the Optuna SOM tuning (labels never used). |
| `quality_metrics.py` | Quality metrics for the NNSOM training experiments. |
| `run_records.py` | Run records for SOM training: run-ID checks, file fingerprints, the run manifest, the never-overwrite check, the deterministic smoke-test subset and the QE history. |
| `som_visualizations.py` | SOM drawing helpers: component planes and feature maps. |
| `train_som.py` ▶ | Trains one SOM on the training embeddings and records the QE after every epoch. |
| `trainer.py` | Training helpers for the NNSOM training experiments. |
| `trainer_history.py` | SOM training with a quantization-error history (the training used for the saved models when ``som.track_qe_history`` is true, as in som.yaml). |
| `visualize_som.py` ▶ | Figures for the selected SOM (SOM Setup with NNSOM). |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
