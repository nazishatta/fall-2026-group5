# Pipeline: How to Run

One Python file runs each step of the project proposal. The runners are grouped by phase, the same way as `tests/` and `outputs/` (`p` = phase), and each is named after the component folder whose scripts it calls, in order.

Run every command from the code root: the folder that holds `src/` and `outputs/` (the layout is the same on EC2 and on a local computer). The runners switch to the code root themselves, so relative paths still work if you start them from elsewhere.

## Step runners, in proposal order

The proposal week is shown for orientation only; the code uses the step names.

| Phase | Step (`run_all.py` name) | Proposal week | Runner | What it does | Run it? |
|---|---|---|---|---|---|
| 1 | `data_pipeline` | 1 | `p1_model_training_baseline/run_data_pipeline.py` | Environment check; MNIST split, class-distribution charts, sample images, `split_stats.json` | ❌ Already done (frozen) |
| 1 | `cnn_baseline_training` | 2 | `p1_model_training_baseline/run_cnn_baseline_training.py` | Train LeNet-5 (metrics, curves, confusion matrix); extract the 84-D embeddings | ❌ Already done (frozen) |
| 2 | `som_nnsom_training` | 3 | `p2_som_training_visualization/run_som_nnsom_training.py` | Train the SOM; SOM figures; 3 convergence charts; grid-size table | ✅ With `--skip-training` to redraw from the trained SOM |
| 2 | `cluster_analysis` | 4 | `p2_som_training_visualization/run_cluster_analysis.py` | BMUs, purity, entropy, class–neuron intersection; class maps, purity map, class pie/stem; cluster heatmaps | ✅ Yes |
| 2 | `error_pattern_analysis` | 5 | `p2_som_training_visualization/run_error_pattern_analysis.py` | Error geography and hotspots; hotspot images; error maps; error heatmaps; error-pattern robustness (15x15, 10x10, 20x20) and verdict | ✅ After `cluster_analysis` |
| 2 | (report) | 3–5 | `p2_som_training_visualization/run_interactive_report.py` | Interactive HTML report | ✅ Yes |
| 4 | `hard_example_mining` | 9 | `p4_guided_retraining/run_hard_example_mining.py` | Decide which training images are hard; SOM, max-softmax and random pool tables (training images only, no CNN trained, test split never loaded) | ✅ Once per run ID; a new `run.revision` in `configs/retraining.yaml` for a new run |
| 4 | `hard_example_mining` (curriculum) | 9 | `p4_guided_retraining/run_curriculum_comparison.py` | Train standard and curriculum arms (5 seeds, validation images only); validation summary; the single test pass after the protocol is frozen | ✅ `--stage train`, then `summary`, then `final_test` |
| 5 | `som_reanalysis` | 11 | `p4_guided_retraining/run_som_reanalysis.py` | Embeddings of the retrained CNNs; new SOM per cell; tables and figures | ✅ `--stage embeddings`, `train_som`, `analyze` |
| 4 | `targeted_augmentation` | 10 | `p4_guided_retraining/run_targeted_augmentation.py` | Check the altered images; train standard and augmented arms; summary; the single test pass | ✅ `--stage check`, `train`, `summary`, `final_test` |
| 4 | (protocol) | 9–10 | `p4_guided_retraining/run_protocol_stamp.py` | Freeze the retraining protocol once every open item is filled | ✅ Once |
| – | – | – | `run_all.py` | Runs the steps above in order (default: `som_nnsom_training` to `error_pattern_analysis`, then the report) | ✅ Yes |

The order is set in `common.py` (`STEP_RUNNERS`). Phase 3 has its own runners (see below); `run_all.py` runs them as the last step, `novelty_detection`, when asked. The Phase 4 runner is standalone for now and is not yet in `STEP_RUNNERS`.

`data_pipeline` and `cnn_baseline_training` made the checkpoint and embeddings that every SOM result uses. Running them again would replace those files and change every result, so their runners refuse. A new CNN run needs a new `run.id` (see `configs/cnn_baseline.yaml`).

## Phase 3 runners

`p3_novelty_detection/` has one runner per component folder and nothing else. Each Phase 3 script has its own config (its own run ID), so these runners take `--only STEP` and `--dry-run` instead of one shared `--config`; their shared code (`P3Step`, `run_phase3`) is in `common.py`, next to the Phase 1 and 2 helpers.

| Runner | Steps |
|---|---|
| `p3_novelty_detection/run_qe_novelty_detection.py` | `qe_novelty`, `qe_score_distribution_figure`, `qe_ood`, `qe_random_control`, `qe_threshold_sweep` |
| `p3_novelty_detection/run_u_matrix_analysis.py` | `qe_u_matrix`, `qe_umatrix_validation_figure`, `u_matrix_grid_figure`, `proposal_gap_analysis`, `svhn_som_overlay` |
| `p3_novelty_detection/run_novelty_benchmark.py` | `svhn_benchmark`, `historical_baseline_verification`, `required_baselines`, `required_baselines_knn`, `qe_msp_paired_comparison`, `svhn_score_export`, `qe_energy_knn_paired`, `qe_umatrix_random_ood`, `standardized_autoencoder`, `fair_budget_baselines`, `unified_novelty_benchmark`, `svhn_inverted_norm_diagnostic`, `qe_norm_dependence`, `autoencoder_scaling_diagnostic`, `missed_svhn_figure`, `historical_svhn_inspection`, `consolidate_evidence` |

`run_all.py --to novelty_detection` (or `--from novelty_detection` for Phase 3 alone) runs the three runners in the order their results depend on each other, defined by `ORDER` in `run_all.py`: `qe_novelty` and `qe_score_distribution_figure`, then the whole `run_novelty_benchmark.py` (its `svhn_benchmark` writes the SVHN score arrays that later steps read), then `qe_threshold_sweep`, then the whole `run_u_matrix_analysis.py`. It passes each runner `--steps`, which skips any step whose results exist. `qe_ood`, `qe_random_control` and `qe_u_matrix` only repeat historical results (the `reorg_…` runs already reproduced them), so they run only with `--with-reproductions`.

A step whose results folder (output folder plus `run_id`, from its config) already exists is skipped; results are never overwritten. To repeat a step, copy its config, give it a new `run_id`, and run `--only STEP --config NEW_CONFIG`.

```bash
python src/v1_mnist/component/pipeline/p3_novelty_detection/run_novelty_benchmark.py --dry-run
python src/v1_mnist/component/pipeline/p3_novelty_detection/run_qe_novelty_detection.py --only qe_ood --dry-run
```

## Standalone runners (not in `run_all.py`)

These steps each have their own config and run ID and are run by hand, in the order shown, after the steps above.

| Phase | Runner | Config | What it does |
|---|---|---|---|
| 2 | `p2_som_training_visualization/run_error_pattern_baselines.py` | `error_pattern_baselines.yaml` | k-means cells, kNN impurity and max-softmax against the SOM's error hotspots (`--stage som`, `kmeans`, `final`) |
| 2 | `p2_som_training_visualization/run_error_detector_budget.py` | `error_detector_budget.yaml` | Same-budget comparison: mistakes caught when every score flags the same share of images (exploratory; reads the baseline run) |
| 2 | `p2_som_training_visualization/run_feature_layer_ablation.py` | `feature_layer_ablation.yaml` | Feature-layer ablation: SOMs on fc1 and fc2, 5 SOM seeds each (`--stage extract`, `som`, `compare`) |
| 2 | `component/p2_som_training_visualization/som_nnsom_training/optuna_grid_tuning.py` (in the component folder, not in `pipeline/`) | `som_optuna.yaml` | Optuna tuning of the three grids and the 5-seed confirmation |
| 3 | `p3_novelty_detection/run_held_out_class_novelty.py` | `held_out_class_novelty.yaml` | Near novelty: train on 6 digits, the other 4 are novel, 5 class splits (`--stage run`, then `compare`) |
| 4 | `p4_guided_retraining/run_som_reanalysis.py --config configs/som_reanalysis_msp_arm.yaml` | `som_reanalysis_msp_arm.yaml` | Secondary SOM re-analysis with the least-confident (max-softmax) arm |
| 4 | `p4_guided_retraining/run_novelty_after_retraining.py` | `novelty_after_retraining.yaml` | Phase 3's novelty scores on the retrained networks (`--stage score`, then `compare`) |

## One pipeline, any dataset

The runners hold no dataset. The dataset comes from the folder of configs: every config merges the `base.yaml` in its own folder, which names the dataset (`dataset.name`, classes, image size) and the results folder (`project.output_root`). `run_all.py --config-dir DIR` runs every step with the configs in `DIR` (default: the MNIST folder, `src/v1_mnist/component/configs`); the Phase 3 runners take `--config-dir` too.

- MNIST: `python src/v1_mnist/component/pipeline/run_all.py` (or `bash src/v1_mnist/shellscripts/run_pipeline.sh`)
- Fashion-MNIST: `python src/v1_fashion_mnist/component/pipeline/run_all.py --from data_pipeline` (or `bash src/v1_fashion_mnist/shellscripts/run_pipeline.sh`), which calls this `run_all.py` with `--config-dir src/v1_fashion_mnist/component/configs`

A new dataset needs a config folder with its own `base.yaml` (and, if torchvision has not got it yet, one entry in `utils/image_datasets.py`).

## Options every runner has

| Option | Meaning |
|---|---|
| `--config CFG` | Config file. `data_pipeline` and `cnn_baseline_training` default to `configs/cnn_baseline.yaml`, the SOM steps to `configs/som.yaml` (the 15x15 headline SOM). Use `configs/som_10x10_robustness.yaml` or `som_20x20_robustness.yaml` for the other grids. |
| `--dry-run` | Print the steps and commands without running anything. |

There is no option to overwrite results (project rule: never overwrite a run). A runner whose results already exist stops before its first step and lists them; to redo that step, move or delete those results first, or use a new run ID. Figures and the interactive report are visualizations and are simply redrawn.

Extra options:

| Runner | Options |
|---|---|
| `run_som_nnsom_training.py` | `--run-id` (also used for the figures and convergence charts), `--skip-training` |
| `run_error_pattern_analysis.py` | `--robustness-configs`, `--data-root` |
| `run_interactive_report.py` | `--split` |
| `run_all.py` | `--config-dir DIR`, `--from STEP`, `--to STEP`, `--cnn-config`, `--som-config`, `--skip-training`, `--no-report`, `--with-reproductions` |

## Common commands

```bash
# See every step without running anything
python src/v1_mnist/component/pipeline/run_all.py --from data_pipeline --dry-run

# Redraw the SOM figures from the trained 15x15 SOM, then run the cluster and
# error-pattern analyses and the report
python src/v1_mnist/component/pipeline/run_all.py --skip-training

# Only the two analyses
python src/v1_mnist/component/pipeline/run_all.py --from cluster_analysis --to error_pattern_analysis

# Phase 3 after the report (steps whose results exist are skipped), or only Phase 3
python src/v1_mnist/component/pipeline/run_all.py --to novelty_detection --dry-run
python src/v1_mnist/component/pipeline/run_all.py --from novelty_detection --dry-run

# One step at a time
python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_som_nnsom_training.py --skip-training
python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_cluster_analysis.py
python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_error_pattern_analysis.py
python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_interactive_report.py

# Train a new SOM under a new run ID (an existing run ID is never retrained)
python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_som_nnsom_training.py --run-id som_15x15_nb11_ep250_s42_v3
```

Each step stops if its key results already exist. `error_pattern_analysis` keeps robustness results that already exist and skips those steps (it says so), because they come from the same code and data.

## Running one script on its own

Every step is still a script you can run directly, for example:

```bash
python src/v1_mnist/component/p2_som_training_visualization/cluster_analysis/cluster_figures.py
```

Each analysis has its own figure script: `som_nnsom_training/visualize_som.py`, `cluster_analysis/cluster_figures.py` and `error_pattern_analysis/error_figures.py`. Each writes its log to its own folder (`logs/`). The heatmaps are drawn the same way: `cluster_analysis/cluster_heatmaps.py` and `error_pattern_analysis/error_heatmaps.py`.

## Where the results go

Everything goes under `outputs/v1_mnist/`:

| Step | Folder |
|---|---|
| `data_pipeline` | `p1_model_training_baseline/data_pipeline/` |
| `cnn_baseline_training` | `p1_model_training_baseline/cnn_baseline_training/` |
| `som_nnsom_training` | `p2_som_training_visualization/som_nnsom_training/` (convergence charts: its `convergence_plots/` folder) |
| `cluster_analysis` | `p2_som_training_visualization/cluster_analysis/` |
| `error_pattern_analysis` | `p2_som_training_visualization/error_pattern_analysis/` |
| (report) | `p2_som_training_visualization/interactive_report/interactive_report.html` |

`demo/` is a hand-picked showcase that is kept in git. Nothing writes to it automatically: copy the figures or charts you want to show there yourself.
