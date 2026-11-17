# MNIST results (`outputs/v1_mnist/`)

Results are grouped by **project phase** and, inside each phase, by **step**, in the order of the project proposal. The folder names match `src/v1_mnist/component/`, `tests/` and `docs/`, so each step's code, tests, documents and results share one name. The paper-writing weeks (12–14) produce documents, not results, so they have no folder here.

> **Naming:** the `p` in `p1_…`, `p2_…`, `p3_…`, `p4_…` means **phase**. Inside a step folder, each run has its own folder named by its **run ID** (for example `som_15x15_nb11_ep250_s42_v2`: 15x15 grid, neighbourhood 11, 250 epochs, seed 42, version 2). A run folder is never overwritten; a new run gets a new ID.

## Folders in proposal order

| Phase | Proposal week | Folder | What is inside |
|---|---|---|---|
| **1: Model Training Baseline** | 1 | [`p1_model_training_baseline/data_pipeline/`](p1_model_training_baseline/data_pipeline/README.md) | Split sizes, class-distribution charts, sample images |
| | 2 | [`p1_model_training_baseline/cnn_baseline_training/`](p1_model_training_baseline/cnn_baseline_training/README.md) | LeNet-5 checkpoint, 84-number embeddings of every split, training curves, test metrics, logs |
| **2: SOM Training & Visualization** | 3 | [`p2_som_training_visualization/som_nnsom_training/`](p2_som_training_visualization/som_nnsom_training/README.md) | Trained SOMs, run logs, `run_manifest.csv`, QE histories and curves, SOM figures |
| | 3 | [`p2_som_training_visualization/som_optuna_tuning/`](p2_som_training_visualization/som_optuna_tuning/README.md) | Optuna tuning of the three grid sizes and its 5-seed check |
| | 4 | [`p2_som_training_visualization/cluster_analysis/`](p2_som_training_visualization/cluster_analysis/README.md) | Neuron assignments, purity and entropy tables, class maps |
| | 5 | [`p2_som_training_visualization/error_pattern_analysis/`](p2_som_training_visualization/error_pattern_analysis/README.md) | Error geography, hotspots, robustness on three grids, comparison with simpler error detectors |
| | 3–5 | `p2_som_training_visualization/interactive_report/` (+ `.dvc`) | `interactive_report.html` (DVC) |
| **3: Novelty Detection** | 6 | [`p3_novelty_detection/qe_novelty_detection/`](p3_novelty_detection/qe_novelty_detection/README.md) | QE scores with the frozen training cut-off, QE against SVHN, cut-off sweep, random control |
| | 7 | [`p3_novelty_detection/u_matrix_analysis/`](p3_novelty_detection/u_matrix_analysis/README.md) | U-Matrix statistics and figures, SVHN on the SOM grid |
| | 8 | [`p3_novelty_detection/novelty_benchmark/`](p3_novelty_detection/novelty_benchmark/README.md) | QE and U-Matrix against Isolation Forest, One-Class SVM, autoencoder, max-softmax, energy, Mahalanobis and kNN; paired comparisons; diagnostics |
| | – | [`p3_novelty_detection/held_out_class_novelty/`](p3_novelty_detection/held_out_class_novelty/README.md) | Train on 6 digits, detect the other 4 (5 class splits) |
| **4: Guided Retraining** | 9 | [`p4_guided_retraining/hard_example_mining/`](p4_guided_retraining/hard_example_mining/README.md) | Hard-image pools, curriculum comparison |
| | 10 | [`p4_guided_retraining/targeted_augmentation/`](p4_guided_retraining/targeted_augmentation/README.md) | Retraining with altered copies of the hard images |
| | 11 | [`p4_guided_retraining/som_reanalysis/`](p4_guided_retraining/som_reanalysis/README.md) | New SOMs on the retrained CNNs |
| | – | [`p4_guided_retraining/novelty_after_retraining/`](p4_guided_retraining/novelty_after_retraining/README.md) | SVHN novelty on the retrained networks |

Every Phase 3 run folder, the script and config that wrote it, and how that was checked, are listed in `p3_novelty_detection/run_manifest.csv`.

## What git keeps, and what DVC keeps

- **In git:** small metrics and records (`.json`, `.csv`), the DVC pointer files (`.dvc`), READMEs and `.gitkeep` files. `src/v1_mnist/tests/test_repository_rules.py` checks this.
- **In DVC** (S3 remote `s3://dats-capstone/shared/group5/dvc-cache`, downloaded with `dvc pull`): checkpoints, embeddings, SOM models, neuron assignments, scores, logs, figures and the interactive report. A folder with a `<name>.dvc` file next to it is one DVC item; it has no README of its own because DVC keeps the whole folder as one unit.

## Things to know

- **Official SOM:** `som_15x15_nb11_ep250_s42_v2`. The 10x10 and 20x20 maps are only for the robustness check. `som_15x15_seed42_v1` is an old run; do not use it (see `p2_som_training_visualization/som_nnsom_training/README.md`).
- **The test split** (10,500 images) was used once for the CNN baseline and once in the final test stages of Phase 4 (after the retraining protocol was frozen). Phases 2 and 3 use training and validation images only.
- **The data pipeline log (`prepare_data.log`) is in `cnn_baseline_training/logs/`**, because DVC tracks that folder as one unit with the other two Phase 1 logs.
- **The U-Matrix figure in `som_nnsom_training/`** is the SOM training picture of the map; the U-Matrix novelty scores are in `p3_novelty_detection/u_matrix_analysis/`.
