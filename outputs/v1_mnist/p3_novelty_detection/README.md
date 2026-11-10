# Phase 3: Novelty Detection

> `p3` in the folder name means **Phase 3**.

Flagging out-of-distribution samples with the SOM. Phase 3 novelty detection results include QE, U-Matrix, SVHN benchmark comparisons, random controls, baseline evaluations, uncertainty estimates, and PDF figures. Results are organized into separate immutable run-ID directories. Historical evidence is preserved without overwriting runs.

```
p3_novelty_detection/
├── qe_novelty_detection/          # quantization-error novelty scores
├── u_matrix_analysis/             # U-Matrix novelty scores and figures
└── novelty_benchmark/             # comparison with the other novelty methods
```

## Start here: where to find each result

The proposal's Phase 3 has three steps, and each folder has its own README that lists its files in the proposal's order: 1. [`qe_novelty_detection/`](qe_novelty_detection/README.md) (quantization error), 2. [`u_matrix_analysis/`](u_matrix_analysis/README.md) (U-Matrix and the map), 3. [`novelty_benchmark/`](novelty_benchmark/README.md) (comparison with other methods).

Use this table when you only want a number. The "Date" column in the tables below is the day the folder was last committed, so you can tell which run is newest.

| What you want | Run folder | File |
|---|---|---|
| QE detector vs SVHN (headline AUROC 0.981) | `qe_novelty_detection/mnist_svhn_qe_frozen_s42_initial` | `qe_ood_metrics.json` |
| QE cutoff sweep (90th to 99.5th percentile) | `qe_novelty_detection/mnist_svhn_qe_threshold_sweep_s42_v1` | `qe_threshold_sweep.json` |
| Random control (permutation null) | `qe_novelty_detection/qe_random_control_evaluation_initial` | `qe_random_control_metrics.json` |
| U-Matrix vs QE on validation images (correlation) | `u_matrix_analysis/qe_u_matrix_validation_s42_initial` | `qe_u_matrix_validation_initial.json` |
| U-Matrix detector vs SVHN (AUROC 0.970), and the baselines the proposal names | `novelty_benchmark/mnist_svhn_five_method_comparison_s42_initial` | `comparison_metrics.json` |
| Where SVHN lands on the map | `u_matrix_analysis/mnist_svhn_som_overlay_s42_v1` | `analysis.json`, `tables/` |
| Gap analysis against the proposal | `u_matrix_analysis/mnist_svhn_proposal_gap_analysis_s42_v1` | `analysis.json` |
| Nine-method benchmark | `novelty_benchmark/mnist_svhn_unified_nine_methods_s42_46_v1` | `unified_nine_method_metrics.json` |
| Baselines with the same tuning budget | `novelty_benchmark/mnist_svhn_fair_budget_baselines_s42_46_v1` | `fair_budget_metrics.json` |
| QE vs max-softmax (paired) | `novelty_benchmark/mnist_svhn_qe_msp_reproduction_s42_v1` | `qe_msp_paired_metrics.json` |
| QE vs energy and kNN (paired) | `novelty_benchmark/mnist_svhn_qe_energy_knn_paired_s42_v1` | `paired_comparison_metrics.json` |
| Embedding-norm check on SVHN | `novelty_benchmark/svhn_inverted_embedding_norm_s42_v1` | `embedding_norm_metrics.json` |
| Autoencoder scaling check | `novelty_benchmark/mnist_svhn_autoencoder_scaling_diagnostic_s42_46_v1` | `autoencoder_scaling_metrics.json`, `tables/` |
| All headline numbers in one table | `novelty_benchmark/mnist_svhn_consolidated_evidence_s42_v1` | `phase3_consolidated_metrics.csv` |

**How to read the run folders**

- A folder is never overwritten. A repeat of a step gets a new run ID, so several folders can belong to the same step.
- **Cite the folders in the table above.** The reports and `CONSOLIDATED_EVIDENCE.md` use them.
- **`reorg_20261102_` folders are reproductions.** Five steps were repeated with today's code after the code was reorganised. Their names describe that event, not the content (see `src/v1_mnist/docs/DEVIATIONS.md`). Do not cite them as new results; the table below says which original each one reproduces.
- **"historical" rows** were written before the current configs. Some are still the source of the headline numbers.
- The newest results are the three analyses run on EC2 on 2026-11-03: the cutoff sweep, the SVHN overlay and the autoencoder scaling check.

## Which code wrote which folder

Every run folder is listed in [`run_manifest.csv`](run_manifest.csv), one row per folder, like Phase 2's `run_manifest.csv`. The tables below show the same rows. A test (`src/v1_mnist/tests/p3_novelty_detection/test_run_manifest.py`) checks that every folder has a row, that every script and config named here exists, and that recorded config hashes still match.

Columns:

- **Script** is in `src/v1_mnist/component/p3_novelty_detection/<step folder>/`; **config** is in `src/v1_mnist/component/configs/`.
- **Config check** says how we know which config made the folder. The strongest case is "recorded config SHA-256 equals this file": the result stores a fingerprint of its config, and today's file has the same fingerprint.
- **Role**: *result* is what the reports cite; *reproduction* is a repeat run made with today's code, kept because it is the only link between an older result and today's code; *historical* is an early run with no surviving code.
- Folder names are never changed, because each result file records the run ID it was written under.

### `qe_novelty_detection/` (QE novelty scores; runner `run_qe_novelty_detection.py`)

| Run folder | Role | Last commit | Step and script | Config | Config check | Used in |
|---|---|---|---|---|---|---|
| `mnist_svhn_qe_frozen_s42_initial` | result | 2026-10-12 | `qe_ood`, `run_qe_ood.py` | - | no config hash recorded; the original config was not found in git history; reproduced with identical numbers by reorg_20261102_qe_ood_s42_v2 | report tab:svhn_new and tab:unified (QE row); docs/DEVIATIONS.md; docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/qe_novelty_detection/HISTORICAL_PROVENANCE_AUDIT.md |
| `mnist_svhn_qe_threshold_sweep_s42_v1` | result | 2026-11-03 | `qe_threshold_sweep`, `run_qe_threshold_sweep.py` | `qe_threshold_sweep.yaml` | recorded config SHA-256 equals this file | docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/novelty_benchmark/CONSOLIDATED_EVIDENCE.md |
| `qe_random_control_evaluation_initial` | result | 2026-10-12 | `qe_random_control`, `run_qe_random_control.py` | - | no config hash recorded; the original config was not found in git history; reproduced by reorg_20261102_random_control_s42_v2 (summary values match; six per-seed counts differ) | docs/DEVIATIONS.md; docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/qe_novelty_detection/HISTORICAL_PROVENANCE_AUDIT.md; docs/p3_novelty_detection/qe_novelty_detection/RANDOM_CONTROL_REPRODUCTION.md |
| `qe_training_calibrated_initial` | result | 2026-10-12 | `qe_novelty`, `run_qe_novelty.py` | `qe_novelty.yaml` | no config hash recorded; the run ID matches this config | report fig:qe (qe_score_distribution.pdf, written by run_qe_novelty.py); docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/qe_novelty_detection/HISTORICAL_PROVENANCE_AUDIT.md |
| `reorg_20261102_qe_novelty_s42_v2` | reproduction | 2026-11-02 | `qe_novelty`, `run_qe_novelty.py` | `phase3_reorganization/reorg_20261102/qe_novelty_rerun.yaml` | no config hash recorded; the run ID matches this config | reproduces qe_training_calibrated_initial (identical numbers); input to the other reproductions |
| `reorg_20261102_qe_ood_s42_v2` | reproduction | 2026-11-02 | `qe_ood`, `run_qe_ood.py` | `phase3_reorganization/reorg_20261102/qe_ood_rerun.yaml` | no config hash recorded; the run ID matches this config | reproduces mnist_svhn_qe_frozen_s42_initial (identical numbers) |
| `reorg_20261102_random_control_s42_v2` | reproduction | 2026-11-02 | `qe_random_control`, `run_qe_random_control.py` | `phase3_reorganization/reorg_20261102/random_control_rerun.yaml` | recorded config SHA-256 equals this file | reproduces qe_random_control_evaluation_initial (six per-seed counts differ) |

### `u_matrix_analysis/` (U-Matrix analysis and figures; runner `run_u_matrix_analysis.py`)

| Run folder | Role | Last commit | Step and script | Config | Config check | Used in |
|---|---|---|---|---|---|---|
| `mnist_qe_umatrix_validation_figure_s42_v1` | result | 2026-10-20 | `qe_umatrix_validation_figure`, `run_qe_umatrix_validation_figure.py` | `qe_umatrix_validation_figure.yaml` | recorded config SHA-256 equals this file | docs/p3_novelty_detection/README.md |
| `mnist_svhn_proposal_gap_analysis_s42_v1` | result | 2026-11-02 | `proposal_gap_analysis`, `run_proposal_gap_analysis.py` | `phase3_proposal_gap_analysis.yaml` | recorded config SHA-256 equals this file | docs/DEVIATIONS.md; docs/p3_novelty_detection/README.md |
| `mnist_svhn_som_overlay_s42_v1` | result | 2026-11-03 | `svhn_som_overlay`, `run_svhn_som_overlay.py` | `svhn_som_overlay.yaml` | recorded config SHA-256 equals this file | docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/novelty_benchmark/CONSOLIDATED_EVIDENCE.md |
| `mnist_umatrix_hex_grid_s42_v1` | result | 2026-10-20 | `u_matrix_grid_figure`, `run_u_matrix_grid_figure.py` | `u_matrix_grid_figure.yaml` | recorded config SHA-256 equals this file | docs/p3_novelty_detection/README.md |
| `qe_u_matrix_validation_s42_initial` | result | 2026-10-19 | `qe_u_matrix`, `run_qe_u_matrix.py` | - | no config hash recorded; the original config was not found in git history; reproduced with identical statistics by reorg_20261102_u_matrix_s42_v2 | docs/DEVIATIONS.md; docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/qe_novelty_detection/HISTORICAL_PROVENANCE_AUDIT.md |
| `reorg_20261102_u_matrix_s42_v2` | reproduction | 2026-11-02 | `qe_u_matrix`, `run_qe_u_matrix.py` | `phase3_reorganization/reorg_20261102/u_matrix_rerun.yaml` | recorded config SHA-256 equals this file | reproduces qe_u_matrix_validation_s42_initial (identical statistics) |

### `novelty_benchmark/` (comparison with the other novelty methods; runner `run_novelty_benchmark.py`)

| Run folder | Role | Last commit | Step and script | Config | Config check | Used in |
|---|---|---|---|---|---|---|
| `mnist_historical_baseline_verification_s42_v1` | result | 2026-10-19 | `historical_baseline_verification`, `run_historical_baseline_verification.py` | `historical_baseline_verification.yaml` | no config hash recorded; the run ID matches this config | docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/qe_novelty_detection/HISTORICAL_PROVENANCE_AUDIT.md |
| `mnist_svhn_ae_train_standardized_s42_v1` | result | 2026-10-20 | `standardized_autoencoder`, `run_standardized_autoencoder.py` | `autoencoder_standardized_diagnostic.yaml` | recorded config SHA-256 equals this file | docs/DEVIATIONS.md; docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/novelty_benchmark/BASELINE_SCORE_ANALYSIS.md |
| `mnist_svhn_autoencoder_scaling_diagnostic_s42_46_v1` | result | 2026-11-03 | `autoencoder_scaling_diagnostic`, `run_autoencoder_scaling.py` | `autoencoder_scaling_diagnostic.yaml` | recorded config SHA-256 equals this file | docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/novelty_benchmark/BASELINE_SCORE_ANALYSIS.md; docs/p3_novelty_detection/novelty_benchmark/CONSOLIDATED_EVIDENCE.md |
| `mnist_svhn_consolidated_evidence_s42_v1` | result | 2026-10-20 | `consolidate_phase3_evidence.py` (own command) | - | no config: inputs are the run folders named in phase3_consolidation_provenance.json | docs/p3_novelty_detection/README.md |
| `mnist_svhn_energy_knn_score_export_s42_v1` | result | 2026-11-02 | `svhn_score_export`, `run_required_baselines.py` | `svhn_score_export.yaml` | recorded config SHA-256 equals this file | docs/DEVIATIONS.md; docs/p3_novelty_detection/README.md |
| `mnist_svhn_fair_budget_baselines_s42_46_v1` | result | 2026-10-20 | `fair_budget_baselines`, `run_fair_budget_baselines.py` | `fair_budget_baselines.yaml` | recorded config SHA-256 equals this file | docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/novelty_benchmark/BASELINE_SCORE_ANALYSIS.md; docs/p3_novelty_detection/novelty_benchmark/CONSOLIDATED_EVIDENCE.md |
| `mnist_svhn_five_method_comparison_s42_initial` | result | 2026-10-12 | `svhn_benchmark`, `run_svhn_benchmark.py` | `novelty_benchmark.yaml` | recorded config SHA-256 equals this file at commit 7411c60; the file was edited later; reproduced with current code by reorg_20261102_svhn_score_export_s42_v1 (14 of 15 table values equal; U-Matrix FPR@95%TPR 0.0701 vs 0.0703) | report tab:svhn (all 15 values match); docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/novelty_benchmark/BASELINE_SCORE_ANALYSIS.md; docs/p3_novelty_detection/novelty_benchmark/CONSOLIDATED_EVIDENCE.md; docs/p3_novelty_detection/novelty_benchmark/SVHN_BASELINE_DIAGNOSTIC.md |
| `mnist_svhn_missed_qe_figure_s42_v1` | result | 2026-10-20 | `missed_svhn_figure`, `run_missed_svhn_figure.py` | `missed_svhn_figure.yaml` | recorded config SHA-256 equals this file | docs/p3_novelty_detection/README.md |
| `mnist_svhn_qe_energy_knn_paired_s42_v1` | result | 2026-10-20 | `qe_energy_knn_paired`, `run_qe_energy_knn_paired.py` | `qe_energy_knn_paired.yaml` | recorded config SHA-256 equals this file | docs/p3_novelty_detection/README.md |
| `mnist_svhn_qe_msp_reproduction_s42_v1` | result | 2026-10-19 | `qe_msp_paired_comparison`, `run_required_baselines.py` | `svhn_paired_comparison.yaml` | recorded config SHA-256 equals this file | docs/DEVIATIONS.md; docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/novelty_benchmark/CONSOLIDATED_EVIDENCE.md |
| `mnist_svhn_qe_umatrix_random_s42_v1` | result | 2026-10-20 | `qe_umatrix_random_ood`, `run_qe_umatrix_random_ood.py` | `qe_u_matrix_random_ood.yaml` | recorded config SHA-256 equals this file | docs/p3_novelty_detection/README.md |
| `mnist_svhn_required_baselines_knn_s42_v1` | result | 2026-10-19 | `required_baselines_knn`, `run_required_baselines.py` | `svhn_knn_baselines.yaml` | recorded config SHA-256 equals this file | report tab:svhn_new (energy, MSP, kNN, Mahalanobis rows); docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/novelty_benchmark/CONSOLIDATED_EVIDENCE.md |
| `mnist_svhn_required_baselines_reproduction_initial` | result | 2026-10-19 | `required_baselines`, `run_required_baselines.py` | `required_ood_baselines.yaml` | recorded config SHA-256 equals this file | docs/p3_novelty_detection/README.md |
| `mnist_svhn_unified_nine_methods_s42_46_v1` | result | 2026-10-27 | `unified_novelty_benchmark`, `unified_novelty_benchmark.py` | `unified_novelty_benchmark.yaml` | recorded config SHA-256 equals this file | report tab:unified; docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/novelty_benchmark/BASELINE_SCORE_ANALYSIS.md |
| `svhn_inverted_embedding_norm_s42_v1` | result | 2026-10-27 | `svhn_inverted_norm_diagnostic`, `svhn_inverted_norm_diagnostic.py` | `svhn_inverted_norm_diagnostic.yaml` | recorded config SHA-256 equals this file | docs/p3_novelty_detection/README.md; docs/p3_novelty_detection/novelty_benchmark/BASELINE_SCORE_ANALYSIS.md |
| `reorg_20261102_svhn_score_export_s42_v1` | reproduction | 2026-11-02 | `svhn_benchmark`, `run_svhn_benchmark.py` | `phase3_reorganization/reorg_20261102/svhn_score_export.yaml` | recorded config SHA-256 equals this file | reproduces mnist_svhn_five_method_comparison_s42_initial with current code; holds the per-method score arrays (DVC) |
| `autoencoder_mnist_validation_s42_initial` | historical | 2026-10-19 | - | - | no script in the repository or its git history contains this file name; its numbers are recomputed by mnist_historical_baseline_verification_s42_v1 | docs/p3_novelty_detection/README.md |
| `classical_mnist_validation_s42_initial` | historical | 2026-10-19 | - | - | no script in the repository or its git history contains this file name; its numbers are recomputed by mnist_historical_baseline_verification_s42_v1 | docs/p3_novelty_detection/README.md |
| `mnist_validation_reproducibility_s42_initial` | historical | 2026-10-19 | - | - | no script in the repository or its git history contains this file name | docs/p3_novelty_detection/README.md |

**Configs with no folder yet.** `qe_ood_evaluation.yaml` (`mnist_svhn_qe_dev_initial`), `qe_random_control.yaml` (`qe_random_control_reproduction_initial`) and `qe_u_matrix_reproduction.yaml` (`qe_u_matrix_validation_reproduction_initial`) are the configs the runners use for steps `qe_ood`, `qe_random_control` and `qe_u_matrix`. They have not been run, because the `reorg_20261102_` reproductions above already repeat these steps. `run_all.py` skips them unless `--with-reproductions` is given.

Large `.npy` outputs are tracked with DVC and are not committed to git; small `.csv` and `.json` files stay in git. Figures are PDF only.
