# Configs (`src/v1_mnist/component/configs/`)

Every setting lives here, never in the code: paths, run IDs, seeds, grid sizes, thresholds and hyperparameters. `base.yaml` is read by every step. Each runner says which config it reads; for a step with several configs (for example the SOM robustness maps) each config makes its own run ID. Configs that were used for a finished run are not edited; a change means a new config or a new run ID.

## Folders

| Folder | What is inside |
|---|---|
| [`dataset/`](dataset/README.md) | Facts about the dataset, read by the data pipeline and through `utils/dataset_info.py`. |
| [`phase3_reorganization/`](phase3_reorganization/README.md) | Configs used to re-run Phase 3 results under new run IDs, one folder per reorganisation date. |

## Files

| File | What it is |
|---|---|
| `autoencoder_scaling_diagnostic.yaml` | Phase 3 diagnostic: does the autoencoder baseline change when its inputs are scaled differently? |
| `autoencoder_standardized_diagnostic.yaml` | Phase 3 diagnostic: the autoencoder baseline on standardised embeddings. |
| `base.yaml` | Shared settings read by every step: dataset and model facts, paths and the seed. |
| `cnn_baseline.yaml` | Phase 1: data split and LeNet-5 training settings. |
| `error_detector_budget.yaml` | Phase 2: same-budget comparison of the error detectors (exploratory). |
| `error_pattern_baselines.yaml` | Phase 2: SOM error cells against k-means cells, kNN impurity and max-softmax. |
| `fair_budget_baselines.yaml` | Phase 3: the classical baselines re-run with the same training budget over seeds 42 to 46. |
| `feature_layer_ablation.yaml` | Phase 2: feature-layer ablation, SOMs on fc1 and fc2 with the same recipe and 5 SOM seeds. |
| `held_out_class_novelty.yaml` | Phase 3: held-out-digit novelty test (6 known digits, 4 new, 5 splits). |
| `historical_baseline_verification.yaml` | Phase 3: re-checks the early baseline results on validation data; old result files stay read-only. |
| `historical_svhn_inspection.yaml` | Phase 3: reads the scores of the early five-method SVHN run for inspection. |
| `missed_svhn_figure.yaml` | Phase 3: figure of the SVHN images that QE misses. |
| `novelty_after_retraining.yaml` | Phase 4: Phase 3's novelty scores on the retrained networks. |
| `novelty_benchmark.yaml` | Phase 3: the classical novelty benchmark (Isolation Forest, One-Class SVM, autoencoder). |
| `phase3_evidence_consolidation.yaml` | Phase 3: inputs and checks for joining the headline Phase 3 results into one table. |
| `phase3_proposal_gap_analysis.yaml` | Phase 3: checks the Phase 3 work against the proposal. |
| `phase3_shared_provenance.yaml` | Phase 3: the frozen inputs (checkpoint, embeddings, SVHN file) and their SHA-256 fingerprints. |
| `qe_energy_knn_paired.yaml` | Phase 3: paired bootstrap comparison of QE, energy and kNN on the same images. |
| `qe_norm_dependence.yaml` | Phase 3: is part of the QE result just the length of the embedding? |
| `qe_novelty.yaml` | Phase 3: QE novelty cut-off set from training images only. |
| `qe_ood_evaluation.yaml` | Phase 3: QE on MNIST validation against SVHN. |
| `qe_random_control.yaml` | Phase 3: random-selection control for the QE result. |
| `qe_score_distribution_figure.yaml` | Phase 3: redraws the QE score-distribution figure under a new run ID. |
| `qe_threshold_sweep.yaml` | Phase 3: QE at several percentile cut-offs. |
| `qe_u_matrix_random_ood.yaml` | Phase 3: QE and U-Matrix against a random score on SVHN, with bootstrap intervals. |
| `qe_u_matrix_reproduction.yaml` | Phase 3: re-runs the QE and U-Matrix analysis on validation data. |
| `qe_umatrix_validation_figure.yaml` | Phase 3: QE against U-Matrix figure. |
| `required_ood_baselines.yaml` | Phase 3: max-softmax, energy and Mahalanobis baselines on the frozen CNN. |
| `retraining.yaml` | Phase 4: hard-example rules, pools, retraining arms, seeds and augmentation (frozen with the protocol). |
| `som.yaml` | Phase 2: the official 15x15 SOM and the settings of the Phase 2 analyses. |
| `som_10x10_robustness.yaml` | Phase 2: the 10x10 SOM used in the robustness check. |
| `som_20x20_robustness.yaml` | Phase 2: the 20x20 SOM used in the robustness check. |
| `som_optuna.yaml` | Phase 2: Optuna tuning of the SOM settings for the 10x10, 15x15 and 20x20 grids. |
| `som_reanalysis.yaml` | Phase 4: new SOMs on the retrained CNNs' embeddings and the map comparison. |
| `som_reanalysis_msp_arm.yaml` | Phase 4: the same re-analysis with the max-softmax arm added. |
| `svhn_inverted_norm_diagnostic.yaml` | Phase 3 diagnostic: SVHN images with inverted colours, to test the embedding-length effect. |
| `svhn_knn_baselines.yaml` | Phase 3: kNN baseline on SVHN. |
| `svhn_paired_comparison.yaml` | Phase 3: paired comparison of QE and max-softmax on SVHN. |
| `svhn_score_export.yaml` | Phase 3: saves the energy and kNN scores of every image for later comparisons. |
| `svhn_som_overlay.yaml` | Phase 3: where the SVHN images land on the SOM. |
| `u_matrix_grid_figure.yaml` | Phase 3: U-Matrix figure on the hexagonal grid. |
| `unified_novelty_benchmark.yaml` | Phase 3: all novelty detectors in one run (exploratory). |
