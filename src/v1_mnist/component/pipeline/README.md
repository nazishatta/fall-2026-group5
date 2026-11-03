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
| 4 | `targeted_augmentation` | 10 | `p4_guided_retraining/run_targeted_augmentation.py` | Check the altered images; train standard and augmented arms; summary; the single test pass | ✅ `--stage check`, `train`, `summary`, `final_test` |
| 4 | (protocol) | 9–10 | `p4_guided_retraining/run_protocol_stamp.py` | Freeze the retraining protocol once every open item is filled | ✅ Once |
| – | – | – | `run_all.py` | Runs the steps above in order (default: `som_nnsom_training` to `error_pattern_analysis`, then the report) | ✅ Yes |

The order is set in `common.py` (`STEP_RUNNERS`). `p3_novelty_detection/` gets its runners when that code exists. The Phase 4 runner is standalone for now and is not yet in `STEP_RUNNERS`.

`data_pipeline` and `cnn_baseline_training` made the checkpoint and embeddings that every SOM result uses. Running them again would replace those files and change every result, so their runners refuse. A new CNN run needs a new `run.id` (see `configs/cnn_baseline.yaml`).

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
| `run_error_pattern_analysis.py` | `--robustness-configs`, `--mnist-root` |
| `run_interactive_report.py` | `--split` |
| `run_all.py` | `--from STEP`, `--to STEP`, `--cnn-config`, `--som-config`, `--skip-training`, `--no-report` |

## Common commands

```bash
# See every step without running anything
python src/v1_mnist/component/pipeline/run_all.py --from data_pipeline --dry-run

# Redraw the SOM figures from the trained 15x15 SOM, then run the cluster and
# error-pattern analyses and the report
python src/v1_mnist/component/pipeline/run_all.py --skip-training

# Only the two analyses
python src/v1_mnist/component/pipeline/run_all.py --from cluster_analysis --to error_pattern_analysis

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
