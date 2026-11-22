# Grid comparison (`src/v1_mnist/component/p2_som_training_visualization/som_nnsom_training/grid_comparison/`)

The pre-registered comparison of the 10x10, 15x15 and 20x20 maps: do their cells predict which validation images the CNN gets wrong, on images not used to estimate the cells' error rates? Primary: 20x20 minus 15x15; secondary: average precision, a low-confidence target, and 15x15 minus 10x10; exploratory (added 2026-11-21, after the MNIST result): 20x20 minus 10x10.

Runner: `pipeline/p2_som_training_visualization/run_grid_comparison.py` (`--stage tune`, `choose`, `boot`, `analyze`, `figure`, or `all`; MNIST also `verify_saved`, `verify_retrain`). Config: `configs/grid_comparison.yaml`. Rules: `docs/p2_som_training_visualization/som_nnsom_training/PREREGISTRATION_grid_comparison.md`. Results: `outputs/v1_mnist/p2_som_training_visualization/grid_comparison/`.

The code was first run on 2026-10-06 outside the repository and brought in on 2026-11-21 with its maths unchanged. On the original run's 30 saved maps it gives the original's numbers exactly (102 numbers, largest difference 0).

## Files

| File | What it is |
|---|---|
| `analysis.py` | The endpoint over the repeats, results.json, and the two reproduction checks against the original MNIST run. |
| `endpoint.py` | The held-out endpoint (stratified halves, cell error rates, cross-fitted AUROC and average precision), the map measures and the decision rule. |
| `fast_trainer.py` | Fast batch SOM: the algorithm of `train_som_with_history` (a test checks they give the same map), with two speed-ups. |
| `figure.py` | The results figure (PDF). |
| `provenance.py` | SHA-256 of the rules, configs and embeddings at the start of a run, checked by every later stage; loading the data and the low-confidence target. |
| `settings.py` | Settings of the grid comparison: grid_comparison.yaml plus the configs it names. |
| `stages.py` | Stage 1 (three neighbourhoods per grid, choice by validation QE, TE1 and TE1+2) and Stage 2 (bootstrap maps). |
