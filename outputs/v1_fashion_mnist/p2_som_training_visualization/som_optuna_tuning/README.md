# `som_optuna_tuning` results (Fashion-MNIST)

Optuna searched the starting neighbourhood and the number of epochs for each grid size with the same budget (20 trials, validation data only), then each chosen setting was checked against the 3-point setting on 5 new SOM seeds. Protocol: `src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/OPTUNA_TUNING_PROTOCOL.md`, followed unchanged (`src/v1_fashion_mnist/docs/p2_som_training_visualization/SOM_ANALYSIS_PROTOCOL.md`).

| Path | What is inside |
|---|---|
| `optuna_grid_summary.csv` | The best trial of each grid: its setting and its validation quantization and topographic errors |
| `som_optuna_<grid>_t20_s42_v1/` | The search for one grid (10x10, 15x15, 20x20): `trials.csv` (every trial) and `selection.json` (the chosen setting) |
| `som_optuna_confirm_<grid>_seeds5_v1/` | The 5-seed check: `confirmation.json` (Optuna minus 3-point, with 95% intervals and a verdict) and `confirmation_runs.csv` (one row per seed and setting) |

Outcome: Optuna's setting is not better than the 3-point setting on any grid (10x10: 3-point better; 15x15: a trade-off, lower quantization error but higher topographic error; 20x20: no clear difference). The headline map is unchanged. Recorded in the Phase 2 protocol, section 5.
