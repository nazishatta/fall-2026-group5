# Novelty after retraining (`outputs/v1_mnist/p4_guided_retraining/novelty_after_retraining/`)

Does the SOM still detect SVHN after the CNN is retrained? Code: `src/v1_mnist/component/p4_guided_retraining/novelty_after_retraining/`. Protocol: `src/v1_mnist/docs/p4_guided_retraining/novelty_after_retraining/NOVELTY_AFTER_RETRAINING_PROTOCOL.md`.

Run folder `novelty_after_retraining_som_15x15_nb11_ep250_s42_v2/`:

| Path | What is inside |
|---|---|
| `results.json` | The verdicts: each retrained arm against standard training, and QE against the other detectors |
| `metrics/<arm>_seed<k>.json`, `metrics/baseline.json` | AUROC of every detector for one network |
| `tables/qe_minus_detectors.csv`, `tables/cell_scores.csv` | QE minus each detector; the cell-based scores |
| `metadata/inputs.json` | The inputs and their SHA-256 fingerprints |
| `scores.dvc` | Every image's scores (DVC) |

Main result: the SOM still detects SVHN after retraining, but no better than before.
