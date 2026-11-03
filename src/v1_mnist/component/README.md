# Code (`src/v1_mnist/component/`)

The code is arranged by **project phase** and, inside each phase, by **project week**, in the same order as the capstone proposal. `tests/` and `outputs/v1_mnist/` use the same folder names, so each week's code, tests and results line up.

> **Naming:** the `p` in `p1_…`, `p2_…`, `p3_…`, `p4_…` means **phase**.

## How to run

One runner per proposal step lives in `pipeline/`; see [`pipeline/README.md`](pipeline/README.md). For example, from the code root (the folder that holds `src/` and `outputs/`):

```bash
python src/v1_mnist/component/pipeline/run_all.py --from data_pipeline --dry-run   # list every step
python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_cluster_analysis.py
```

## Folders in proposal order

| Phase | Week | Folder | What the code does |
|---|---|---|---|
| **1: Model Training Baseline** | 1 | `p1_model_training_baseline/data_pipeline/` | Load MNIST, stratified 70/15/15 split, class-distribution charts, sample images |
| | 2 | `p1_model_training_baseline/cnn_baseline_training/` | LeNet-5, training, evaluation (accuracy, F1, confusion matrix), fc2 embedding extraction |
| **2: SOM Training & Visualization** | 3 | `p2_som_training_visualization/som_nnsom_training/` | NNSOM training with QE history, quality metrics, SOM figures, convergence charts, grid-size table |
| | 4 | `p2_som_training_visualization/cluster_analysis/` | BMU assignment, purity, entropy, class–neuron intersection (NNSOM), class maps, Week 4 heatmaps |
| | 5 | `p2_som_training_visualization/error_pattern_analysis/` | Error geography, hotspots (one rule, `utils/hotspots.py`) and their images, error maps, Week 5 heatmaps, error-pattern robustness and verdict |
| | 3–5 | `p2_som_training_visualization/interactive_report/` | Interactive HTML report of the SOM |
| **3: Novelty Detection** | 6 | `p3_novelty_detection/qe_novelty_detection/` | Implemented |
| | 7 | `p3_novelty_detection/u_matrix_analysis/` | Implemented |
| | 8 | `p3_novelty_detection/novelty_benchmark/` | Implemented |
| **4: Guided Retraining** | 9 | `p4_guided_retraining/hard_example_mining/` | *(empty)* |
| | 10 | `p4_guided_retraining/targeted_augmentation/` | *(empty)* |
| | 11 | `p4_guided_retraining/som_reanalysis/` | *(empty)* |

Shared by every week:

| Folder | Contents |
|---|---|
| `pipeline/` | One runner per week (`p1_…/`, `p2_…/`), `run_all.py`, `common.py` |
| `configs/` | `base.yaml`, `cnn_baseline.yaml`, `som.yaml`, `som_10x10_robustness.yaml`, `som_20x20_robustness.yaml`, `dataset/` |
| `utils/` | `config.py` (config loader, output folders), `logging.py`, `run_settings.py` (which SOM to analyse + hash check), `som_io.py`, `stats.py` (entropy, Wilson bound, Spearman), `heatmaps.py`, `som_plotting.py` (SOM drawing helpers for every week), `som_text_style.py` (label styling for dense NNSOM plots), `som_neuron_maps.py` (per-neuron class/error statistics and the simple grid map), `som_types.py` (type names for NNSOM objects), `som_figure_setup.py` (loads the SOM and data for the figure scripts), `som_data.py` (loads the embeddings and the train-fitted scaler), `hotspots.py` (the one error-hotspot rule), `dataset_info.py` (dataset and model facts read from the config), `figures.py` (saves every figure as one PDF with embedded fonts) |

Dataset and model facts (number of classes, embedding size, layer name, the names used in figure titles and file names) are set once in `configs/base.yaml` under `dataset:` and `model:`. The Phase 2 code reads them from there through `utils/dataset_info.py` instead of hard-coding MNIST values, and checks the embeddings against `model.feature_dim`. Phase 1 and `hotspot_samples.py` stay MNIST-specific on purpose.

## Files in each week folder

```
component/
├── pipeline/
│   ├── run_all.py, common.py, README.md
│   ├── p1_model_training_baseline/     run_data_pipeline.py, run_cnn_baseline_training.py
│   └── p2_som_training_visualization/  run_som_nnsom_training.py, run_cluster_analysis.py,
│                                       run_error_pattern_analysis.py, run_interactive_report.py
├── p1_model_training_baseline/
│   ├── data_pipeline/                  prepare_data.py ▶, mnist_dataset.py, data_plots.py
│   └── cnn_baseline_training/          train_baseline.py ▶, extract_embeddings.py ▶, lenet5.py,
│                                       extractor.py, evaluation.py, cnn_plots.py, run_ids.py
├── p2_som_training_visualization/
│   ├── som_nnsom_training/             train_som.py ▶, visualize_som.py ▶, grid_ablation.py ▶,
│   │                                   run_records.py (run IDs, manifest, QE history), trainer.py,
│   │                                   trainer_history.py (training with QE history), memory_safe_init.py,
│   │                                   quality_metrics.py, som_visualizations.py (Week 3 drawing),
│   │                                   extended_plots.py (dense Week 3 plots),
│   │                                   convergence_plots/ (3 chart scripts ▶, artifacts.py)
│   ├── cluster_analysis/               run_cluster_analysis.py ▶, cluster_figures.py ▶,
│   │                                   cluster_heatmaps.py ▶, cluster_metrics.py
│   ├── error_pattern_analysis/         error_patterns.py ▶, hotspot_samples.py ▶, error_figures.py ▶,
│   │                                   error_heatmaps.py ▶,
│   │                                   error_pattern_robustness.py ▶,
│   │                                   error_pattern_robustness_verdict.py ▶
│   └── interactive_report/             visualize_som_interactive.py ▶ + report package, templates/
├── p3_novelty_detection/               qe_novelty_detection/, u_matrix_analysis/, novelty_benchmark/
├── p4_guided_retraining/               hard_example_mining/, targeted_augmentation/, som_reanalysis/
├── configs/
└── utils/
```

▶ = a step script that can also be run on its own; the week runners call them in order.

## Adding code for a new week

1. Put the code in that week's folder (for example `p3_novelty_detection/qe_novelty_detection/`).
2. Write its outputs to the matching folder under `outputs/v1_mnist/`.
3. Add its tests to the matching folder under `tests/`.
4. Add a step runner in `pipeline/p<phase>_…/`, named after the component folder, and register it in `pipeline/common.py` (`STEP_RUNNERS`, in proposal order) so `run_all.py` runs it.
