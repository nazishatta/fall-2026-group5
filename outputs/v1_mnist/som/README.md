# SOM Results

## Official model

`som_15x15_nb11_ep250_s42_v2`: 15x15 grid, neighbourhood 11, 250 epochs, seed 42 (settings in `src/v1_mnist/component/configs/som.yaml`).

| What | Where |
|---|---|
| Static figures | `figures/som_15x15_nb11_ep250_s42_v2/` (SVG and PDF) |
| Interactive report | `som_interactive/interactive_report.html` |
| Metrics and logs | `logs/som_15x15_nb11_ep250_s42_v2_*` |
| QE history | `tables/som_15x15_nb11_ep250_s42_v2_qe_history.csv` |
| Model fingerprints (SHA-256) | `logs/run_manifest.csv` |

The `som_20x20_nb15_ep250_s42_v2` and `som_10x10_nb7_ep250_s42_v2` files are used only for the RQ2 robustness check.
The older `som_15x15_seed42_final` is a legacy run and should not be used: its metrics file was overwritten (see `src/v1_mnist/docs/som/DEVIATIONS.md`) and its figures were replaced by the v2 figures.

Why 15x15 was chosen: `src/v1_mnist/docs/som/README.md`.

## Training implementation note

The SOMs are trained with `train_som_with_history` (`track_qe_history: true`): a copy of NNSOM's batch SOM loop that recomputes the winning neurons from the updated weights every epoch. NNSOM 1.8.3's own `som.train` picks the winners from the initial weights in every epoch. Both agree after epoch 1. Initialisation, cluster assignment, quality metrics and plots all use NNSOM. The `actual_backend` entry in the metrics files reflects the NNSOM class that was created; the training loop itself runs on CPU/NumPy.
