# `grid_comparison` results (Fashion-MNIST)

The pre-registered comparison of the 10x10, 15x15 and 20x20 maps, with MNIST's rules unchanged: how well do a map's cells predict which validation images the CNN gets wrong, on images not used to estimate the cells' error rates? Rules: `src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/PREREGISTRATION_grid_comparison.md` (Fashion-MNIST Phase 2 protocol, section 2). Code: `src/v1_mnist/component/p2_som_training_visualization/som_nnsom_training/grid_comparison/`. Config: `src/v1_fashion_mnist/component/configs/grid_comparison.yaml`.

Run folder `grid_comparison_g10_15_20_b10_r200/`:

| Path | What is inside |
|---|---|
| `results.json` | Primary (20x20 minus 15x15), secondary and exploratory (20x20 minus 10x10) comparisons with 95% intervals and verdicts; per-grid held-out AUROC and average precision; map measures |
| `stage1_choice.json`, `tables/stage1.csv` | The neighbourhood chosen per grid from validation QE, TE1 and TE1+2 (labels not used) |
| `tables/endpoint_reps.csv` | Every repeat's scores |
| `records/` | One record per map: settings, fingerprints, validation measures |
| `models/` (+ `models.dvc`) | The 9 Stage 1 and 30 bootstrap maps (DVC) |
| `pdf/grid_comparison_results.pdf` | The results figure (DVC) |
| `metadata/inputs.json` | SHA-256 of the rules, configs and embeddings |
| `logs/` (+ `logs.dvc`) | Run logs (DVC) |

Result (2026-11-21): the primary comparison is decided. 20x20 finds the network's mistakes less well than the
headline 15x15 map: held-out AUROC 0.744 against 0.786, difference -0.042, 95% range -0.077 to -0.008
(**15x15 better**). 15x15 against 10x10 is inconclusive. Exploratory (not part of the decision): 20x20 minus
10x10, -0.061 [-0.100, -0.020], 10x10 better. All numbers and their reading: Fashion-MNIST Phase 2 protocol,
section 5 (`src/v1_fashion_mnist/docs/p2_som_training_visualization/SOM_ANALYSIS_PROTOCOL.md`).
