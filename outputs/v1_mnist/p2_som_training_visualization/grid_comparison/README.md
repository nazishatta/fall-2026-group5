# `grid_comparison` results (MNIST)

The pre-registered comparison of the 10x10, 15x15 and 20x20 maps: how well do a map's cells predict which validation images the CNN gets wrong, on images not used to estimate the cells' error rates? Rules: `src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/PREREGISTRATION_grid_comparison.md`. Results and their reading: `src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/GRID_COMPARISON_RESULTS.md`. Code: `src/v1_mnist/component/p2_som_training_visualization/som_nnsom_training/grid_comparison/`.

| Folder | Role | What is inside |
|---|---|---|
| `grid_comparison_initial/` | the original run (2026-10-06), cited in the report | `results.json`, `endpoint_reps.csv` (every repeat), `run_manifest.csv` (all 39 maps with fingerprints), `stage1_choice.json`; `models/` (+ `models.dvc`: the 39 trained maps, DVC) |
| `grid_comparison_g10_15_20_b10_r200/` | reproduction with the repository code | `reproduction/saved_maps_check.json` (the code on the original's saved maps: every number must match); after a fresh run also `results.json`, `stage1_choice.json`, `records/` (one record per map), `tables/` (Stage 1 choice, every repeat), `pdf/`, `reproduction/retrain_check.json`, `metadata/inputs.json`; `models/` (DVC) |

Main result (original run, reproduced exactly by the repository code): 20x20 and 15x15 do not differ (held-out AUROC 0.630 against 0.629; difference +0.001, 95% interval -0.100 to +0.089: inconclusive). Exploratory, added later: 20x20 minus 10x10, -0.064 [-0.188, +0.050].
