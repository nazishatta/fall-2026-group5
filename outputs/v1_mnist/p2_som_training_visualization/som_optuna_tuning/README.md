# Optuna tuning of the SOM (`outputs/v1_mnist/p2_som_training_visualization/som_optuna_tuning/`)

Optuna searched the starting neighbourhood and the number of epochs for each grid size with the same budget (20 trials, validation data only), then each chosen setting was checked against the earlier 3-point choice on 5 new SOM seeds. Protocol: `src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/OPTUNA_TUNING_PROTOCOL.md`.

| Path | What is inside |
|---|---|
| `optuna_grid_summary.csv` | The best trial of each grid: its setting and its validation QE and topographic errors |
| `som_optuna_<grid>_t20_s42_v1/` | The search for one grid (10x10, 15x15, 20x20): `trials.csv` (every trial) and `selection.json` (the chosen setting) |
| `som_optuna_confirm_<grid>_seeds5_v1/` | The 5-seed check: `confirmation.json` (Optuna minus 3-point, with 95% intervals and a verdict) and `confirmation_runs.csv` (one row per seed and setting) |

Outcome: for 15x15 (the headline map) there was no clear difference, so the headline map is unchanged. For 20x20 the 3-point setting was better; for 10x10 the Optuna setting was better. See `src/v1_mnist/docs/DEVIATIONS.md` (2026-11-17).
