# SOM Results

## Training implementation note

The final model `som_15x15_seed42_final` was trained with `train_som_with_history` (NumPy batch SOM that recomputes winners from the updated weights every epoch, `track_qe_history: true`), not NNSOM's native `som.train`, which uses the initial weights to pick winners in every epoch (NNSOM 1.8.x). Both agree after epoch 1. The `actual_backend: gpu` entry in `logs/som_15x15_seed42_final_metrics.json` reflects the NNSOM class that was instantiated; the training loop itself ran on CPU/NumPy.
