# Pipeline: How to Run

One Python file runs each step of the project proposal. The runners are grouped by phase, the same way as `tests/` and `outputs/` (`p` = phase), and each is named after the component folder whose scripts it calls, in order.

Run every command from the code root: the folder that holds `src/` and `outputs/` (the layout is the same on EC2 and on a local computer). The runners switch to the code root themselves, so relative paths still work if you start them from elsewhere.

## Step runners, in proposal order

The proposal week is shown for orientation only; the code uses the step names.

| Phase | Step (`run_all.py` name) | Proposal week | Runner | What it does | Run it? |
|---|---|---|---|---|---|
| 1 | `data_pipeline` | 1 | `p1_model_training_baseline/run_data_pipeline.py` | Environment check; MNIST split, class-distribution charts, sample images, `split_stats.json` | ❌ Already done (frozen) |
| 1 | `cnn_baseline_training` | 2 | `p1_model_training_baseline/run_cnn_baseline_training.py` | Train LeNet-5 (metrics, curves, confusion matrix); extract the 84-D embeddings | ❌ Already done (frozen) |
| 2 | `som_nnsom_training` | 3 | `p2_som_training_visualization/run_som_nnsom_training.py` | First the 10x10 and 20x20 robustness maps (`som_*_robustness.yaml`) if not trained yet; then train the SOM; SOM figures; 3 convergence charts; grid-size table | ✅ With `--skip-training` to redraw from the trained SOM |
| 2 | `cluster_analysis` | 4 | `p2_som_training_visualization/run_cluster_analysis.py` | BMUs, purity, entropy, class–neuron intersection; class maps, purity map, class pie/stem; cluster heatmaps | ✅ Yes |
| 2 | `error_pattern_analysis` | 5 | `p2_som_training_visualization/run_error_pattern_analysis.py` | Error geography and hotspots; hotspot images; error maps; error heatmaps; error-pattern robustness (15x15, 10x10, 20x20) and verdict | ✅ After `cluster_analysis` |
| 2 | (report) | 3–5 | `p2_som_training_visualization/run_interactive_report.py` | Interactive HTML report | ✅ Yes |
| 2 | `error_pattern_baselines` | – | `p2_som_training_visualization/run_error_pattern_baselines.py --stage all` | k-means cells, kNN impurity and max-softmax against the SOM's error hotspots | ✅ Skipped if its results exist |
| 2 | `error_detector_budget` | – | `p2_som_training_visualization/run_error_detector_budget.py` | Same-budget comparison of the error detectors (exploratory) | ✅ Skipped if its results exist |
| 2 | `grid_comparison` | – | `p2_som_training_visualization/run_grid_comparison.py --stage all` | Pre-registered grid comparison: 10x10, 15x15 and 20x20 maps on predicting held-out CNN mistakes (MNIST also checks the original run) | ✅ Skipped if its results exist |
| 2 | `som_optuna` | – | `component/p2_som_training_visualization/som_nnsom_training/optuna_grid_tuning.py`, `optuna_confirmation.py` | Optuna search for every grid in `som_optuna.yaml`, the 5-seed confirmation of each, the summary table | ✅ Skipped if the summary table exists |
| 2 | `feature_layer_ablation` | – | `p2_som_training_visualization/run_feature_layer_ablation.py` | fc1 against fc2: `--stage extract`, `som`, `compare` | ✅ Skipped if its results exist |
| 3 | `novelty_detection` | – | `p3_novelty_detection/run_*.py` (see below) | The three Phase 3 runners, in the order their results depend on each other | ✅ Finished steps are skipped |
| 3 | `novelty_statements` | – | `component/p3_novelty_detection/novelty_benchmark/run_novelty_statements.py` | The six novelty scores in one run: the proposal's RQ3 and RQ4 (RQ4 against the best of max-softmax, energy, Mahalanobis and kNN), and statements O1 to O4 where the dataset's protocol states them | ✅ Skipped if its results exist |
| 3 | `held_out_class_novelty` | – | `p3_novelty_detection/run_held_out_class_novelty.py` | Near novelty with held-out classes: `--stage run`, then `compare` | ✅ Skipped if its results exist |
| 4 | `guided_retraining` | – | `p4_guided_retraining/run_hard_example_mining.py`, `run_targeted_augmentation.py`, `run_curriculum_comparison.py` | Hard-image pools; transform `check`; both experiments `train` and `summary` (validation only) | ✅ Skipped if its results exist |
| 4 | (protocol, by hand) | – | `p4_guided_retraining/run_protocol_stamp.py --config CONFIG` | Fill in the macro-F1 guard, then freeze the retraining protocol. **`run_all.py` never does this** | ✅ Once, before the next row |
| 4 | `guided_retraining_test` | – | the same two experiment runners, `run_som_reanalysis.py`, `run_novelty_after_retraining.py` | The single test pass and its breakdown; SOM re-analysis (and the max-softmax arm, if configured); novelty after retraining | ✅ Only when asked for by name; refuses without the stamp |
| 4 | `hard_example_mining` | 9 | `p4_guided_retraining/run_hard_example_mining.py` | Decide which training images are hard; SOM, max-softmax and random pool tables (training images only, no CNN trained, test split never loaded) | ✅ Once per run ID; a new `run.revision` in `configs/retraining.yaml` for a new run |
| 4 | `hard_example_mining` (curriculum) | 9 | `p4_guided_retraining/run_curriculum_comparison.py` | Train standard and curriculum arms (5 seeds, validation images only); validation summary; the single test pass after the protocol is frozen | ✅ `--stage train`, then `summary`, then `final_test` |
| 5 | `som_reanalysis` | 11 | `p4_guided_retraining/run_som_reanalysis.py` | Embeddings of the retrained CNNs; new SOM per cell; tables and figures | ✅ `--stage embeddings`, `train_som`, `analyze` |
| 4 | `targeted_augmentation` | 10 | `p4_guided_retraining/run_targeted_augmentation.py` | Check the altered images; train standard and augmented arms; summary; the single test pass | ✅ `--stage check`, `train`, `summary`, `final_test` |
| 4 | (protocol) | 9–10 | `p4_guided_retraining/run_protocol_stamp.py` | Freeze the retraining protocol once every open item is filled | ✅ Once |
| – | – | – | `run_all.py` | Runs the steps above in order (default: `som_nnsom_training` to `error_pattern_analysis`, then the report) | ✅ Yes |

The Phase 1 and core Phase 2 runners are listed in `common.py` (`STEP_RUNNERS`); the later steps, from `error_pattern_baselines` on, are defined in `study_steps.py`; `run_all.py` puts them all in one list, `STEP_NAMES`, in proposal order. The default range is unchanged: it ends at `error_pattern_analysis` and the report. `--to STEP` runs on to a later step, and `--from STEP` alone, for a step after the report, runs just that step.

Two rules decide which of the later steps run. A step whose config is not in the `--config-dir` folder is not part of that dataset's study and is skipped with a message. A step whose results all exist already is skipped too; each step's run folders are found by its own settings loader. A half-finished step is planned, and its script stops at the first part that exists.

The Phase 4 steps around the test split are stop points. `guided_retraining` ends with the validation summaries. A person then fills in the macro-F1 guard (protocol and `retraining.yaml`) and runs `run_protocol_stamp.py --config CONFIG`. Only `--to guided_retraining_test` (or `--from` it) runs the test step, and its `final_test` stages refuse without the stamp.

`data_pipeline` and `cnn_baseline_training` made the checkpoint and embeddings that every SOM result uses. Running them again would replace those files and change every result, so their runners refuse. A new CNN run needs a new `run.id` (see `configs/cnn_baseline.yaml`).

## Phase 3 runners

`p3_novelty_detection/` has one runner per component folder and nothing else. Each Phase 3 script has its own config (its own run ID), so these runners take `--only STEP` and `--dry-run` instead of one shared `--config`; their shared code (`P3Step`, `run_phase3`) is in `common.py`, next to the Phase 1 and 2 helpers.

| Runner | Steps |
|---|---|
| `p3_novelty_detection/run_qe_novelty_detection.py` | `qe_novelty`, `qe_score_distribution_figure`, `qe_ood`, `qe_random_control`, `qe_threshold_sweep` |
| `p3_novelty_detection/run_u_matrix_analysis.py` | `qe_u_matrix`, `qe_umatrix_validation_figure`, `u_matrix_grid_figure`, `proposal_gap_analysis`, `svhn_som_overlay` |
| `p3_novelty_detection/run_novelty_benchmark.py` | `svhn_benchmark`, `historical_baseline_verification`, `required_baselines`, `required_baselines_knn`, `qe_msp_paired_comparison`, `svhn_score_export`, `qe_energy_knn_paired`, `qe_umatrix_random_ood`, `standardized_autoencoder`, `fair_budget_baselines`, `unified_novelty_benchmark`, `svhn_inverted_norm_diagnostic`, `qe_norm_dependence`, `autoencoder_scaling_diagnostic`, `missed_svhn_figure`, `historical_svhn_inspection`, `consolidate_evidence` |

`run_all.py --to novelty_detection` (or `--from novelty_detection` for Phase 3 alone) runs the three runners in the order their results depend on each other, defined by `ORDER` in `run_all.py`: `qe_novelty`, `qe_score_distribution_figure` and `qe_ood` (whose metrics `svhn_benchmark` must reproduce), then the whole `run_novelty_benchmark.py` (its `svhn_benchmark` writes the SVHN score arrays that later steps read), then `qe_random_control` and `qe_threshold_sweep`, then the whole `run_u_matrix_analysis.py`. It passes each runner `--steps`, which skips any step whose results exist. `qe_ood`, `qe_random_control` and `qe_u_matrix` only repeat historical results (the `reorg_…` runs already reproduced them), so they run only with `--with-reproductions`.

A step whose results folder (output folder plus `run_id`, from its config) already exists is skipped; results are never overwritten. To repeat a step, copy its config, give it a new `run_id`, and run `--only STEP --config NEW_CONFIG`.

```bash
python src/v1_mnist/component/pipeline/p3_novelty_detection/run_novelty_benchmark.py --dry-run
python src/v1_mnist/component/pipeline/p3_novelty_detection/run_qe_novelty_detection.py --only qe_ood --dry-run
```

## Running a later step by hand

Every step in `study_steps.py` can still be run alone, with its own config and stage, for example to finish a half-finished run or to repeat a stage under a new run ID:

```bash
python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_feature_layer_ablation.py --config src/v1_mnist/component/configs/feature_layer_ablation.yaml --stage compare
python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_grid_comparison.py --config src/v1_mnist/component/configs/grid_comparison.yaml --stage verify_saved
python src/v1_mnist/component/pipeline/p4_guided_retraining/run_protocol_stamp.py --config src/v1_mnist/component/configs/retraining.yaml
```

## One pipeline, any dataset

The runners hold no dataset. The dataset comes from the folder of configs: every config merges the `base.yaml` in its own folder, which names the dataset (`dataset.name`, classes, image size) and the results folder (`project.output_root`). `run_all.py --config-dir DIR` runs every step with the configs in `DIR` (default: the MNIST folder, `src/v1_mnist/component/configs`); the Phase 3 runners take `--config-dir` too.

- MNIST: `python src/v1_mnist/component/pipeline/run_all.py` (or `bash src/v1_mnist/shellscripts/run_pipeline.sh`)
- Fashion-MNIST: `python src/v1_fashion_mnist/component/pipeline/run_all.py --from data_pipeline` (or `bash src/v1_fashion_mnist/shellscripts/run_pipeline.sh`), which calls this `run_all.py` with `--config-dir src/v1_fashion_mnist/component/configs`

A new dataset needs a config folder with its own `base.yaml` (and, if torchvision has not got it yet, one entry in `utils/image_datasets.py`). For Phase 3 and later, that folder also needs its own `phase3_shared_provenance.yaml` (its checkpoint, embeddings and outside-set file, with their SHA-256): every Phase 3 script checks its inputs against the manifest in its own config folder, so a dataset can never be checked against another dataset's files. Result files and keys that MNIST names after itself (`qe_mnist_val_scores.npy`, `"mnist_validation"`) are named after the dataset (`utils/in_distribution.py`), so MNIST's own names are unchanged.

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
| `run_all.py` | `--config-dir DIR`, `--from STEP`, `--to STEP`, `--cnn-config`, `--som-config`, `--skip-training`, `--no-report`, `--with-reproductions` (MNIST only: for other datasets the three steps are first runs and always planned) |

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

# The whole study up to the stop before the test split, then (after the stamp) the test step
python src/v1_mnist/component/pipeline/run_all.py --to guided_retraining --dry-run
python src/v1_mnist/component/pipeline/run_all.py --from guided_retraining_test --dry-run

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
