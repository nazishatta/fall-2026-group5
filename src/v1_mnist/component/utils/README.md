# Shared code (`component/utils/`)

Helpers used by more than one step of the project. Nothing here runs on its own: every file is imported by the step scripts in `p1_model_training_baseline/`, `p2_som_training_visualization/` and `pipeline/`, or by another file in this folder.

The trees below follow the code's own layout: each **phase folder**, then its **step folders** (one per proposal week), then each script, and under the script the **utils files it calls** with the functions it imports. Paths are relative to `src/v1_mnist/component/`.

## The files

| File                  | What it does                                                                                                                                                                             |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`           | Loads a YAML config with `base.yaml` merged in (`load_config`); sets seeds (`set_seed`, `require_seed`); picks the device (`get_device`). Refuses the old run names `week_2` / `week_3`. |
| `dataset_info.py`     | What the SOM code needs to know about the dataset and the CNN (classes, feature size, display names), read from the config.                                                              |
| `file_hash.py`        | SHA-256 fingerprint of a file (`sha256_file`), used for every provenance check; `som_io.py` re-exports it.                                                                                 |
| `figures.py`          | Saves every results figure the same way: one PDF with embedded fonts (`save_pdf`).                                                                                                       |
| `heatmaps.py`         | Reads per-neuron tables and draws the cluster and error heatmaps.                                                                                                                        |
| `hotspots.py`         | The project's one error-hotspot rule: top max(5, 5% of neurons) by Wilson lower bound.                                                                                                   |
| `logging.py`          | Loggers: `setup_logger` (console and log file), `get_logger`, `get_console_logger` (the runners' plain console output).                                                                  |
| `run_ids.py`          | Refuses run IDs that are empty or contain anything but letters, digits, `_` and `-` (`check_run_id`). Used by the Phase 3 scripts.                                                      |
| `run_settings.py`     | Which SOM the cluster and error-pattern analyses use, where their outputs go, and checks that the model and inputs are the right ones (SHA-256).                                         |
| `som_data.py`         | Loads and checks the CNN embeddings from the configured folder only; fits the scaler on the training split only.                                                                         |
| `som_figure_setup.py` | Shared start-up for the three SOM figure scripts: arguments, config, model, data, figure folders, figure manifest.                                                                       |
| `repo_paths.py`       | Resolves a path from a config inside the repository and refuses `..` or absolute paths that leave it, without following symlinks (`resolve_inside`).                                                     |
| `som_io.py`           | Loads the selected SOM and one split's embeddings; rebuilds BMU assignments; SHA-256 of files; writes CSV rows.                                                                          |
| `som_neuron_maps.py`  | Per-neuron class and error statistics, and the simple grid map.                                                                                                                          |
| `som_plotting.py`     | Runs an NNSOM plot and saves it (`save_nnsom_plot`); picks the most variable features.                                                                                                   |
| `som_text_style.py`   | Text styling for NNSOM's dense plots (neuron labels on hexagons).                                                                                                                        |
| `som_types.py`        | The type name `SOMModel` for NNSOM objects, used only in type hints.                                                                                                                     |
| `stats.py`            | Statistics: Wilson intervals and lower bounds, normalized entropy, safe mean and median, Spearman correlation.                                                                           |
| `__init__.py`         | Makes the folder a package (no code).                                                                                                                                                    |

## 1. At a glance: which step folders use which utils file

A ✓ means at least one script in that step folder imports the file directly (files that a utils file uses in turn are in section 3).

| utils file            | `data_pipeline` | `cnn_baseline_training` | `som_nnsom_training` | `cluster_analysis` | `error_pattern_analysis` | `interactive_report` |
| --------------------- | --------------- | ----------------------- | -------------------- | ------------------ | ------------------------ | -------------------- |
| `config.py`           | ✓               | ✓                       | ✓                    | ✓                  |                          | ✓                    |
| `dataset_info.py`     |                 |                         | ✓                    | ✓                  |                          | ✓                    |
| `figures.py`          | ✓               | ✓                       | ✓                    |                    | ✓                        |                      |
| `heatmaps.py`         |                 |                         |                      | ✓                  | ✓                        |                      |
| `hotspots.py`         |                 |                         |                      |                    | ✓                        |                      |
| `logging.py`          | ✓               | ✓                       | ✓                    | ✓                  | ✓                        | ✓                    |
| `run_settings.py`     |                 |                         | ✓                    | ✓                  | ✓                        |                      |
| `som_data.py`         |                 |                         | ✓                    |                    |                          | ✓                    |
| `som_figure_setup.py` |                 |                         | ✓                    | ✓                  | ✓                        |                      |
| `som_io.py`           |                 |                         |                      | ✓                  | ✓                        |                      |
| `som_neuron_maps.py`  |                 |                         |                      | ✓                  | ✓                        |                      |
| `som_plotting.py`     |                 |                         | ✓                    | ✓                  | ✓                        | ✓                    |
| `som_types.py`        |                 |                         | ✓                    | ✓                  | ✓                        | ✓                    |
| `stats.py`            |                 | ✓                       |                      | ✓                  | ✓                        |                      |

## 2. Utils files that call other utils files

When a script calls one of the files on the left, these are used as well:

```
run_settings.py
├── config.py
├── dataset_info.py
├── som_data.py
└── som_io.py

som_figure_setup.py
├── config.py
├── dataset_info.py
├── som_data.py
└── som_types.py

som_neuron_maps.py
├── som_plotting.py
│   ├── figures.py
│   ├── som_text_style.py
│   └── som_types.py
└── som_types.py

heatmaps.py
└── figures.py

hotspots.py
└── stats.py
```

`config.py`, `dataset_info.py`, `figures.py`, `logging.py`, `som_data.py`, `som_io.py`, `som_text_style.py`, `som_types.py` and `stats.py` use nothing else in this folder.

## Keeping this file current

When you add a script to a step folder, add a utils file, or start using a utils file from a new script, update the trees and the table above. They were made from the actual `import` lines in the code (2026-10-10).

## Every file in this folder

One line per file, taken from the file's own description. The trees above show which step uses which file.

| File | What it is |
|---|---|
| `cnn_retraining.py` | Train LeNet-5 from scratch with the Phase 1 recipe, on any list of images, without the test split. |
| `config.py` | Reads the YAML configs and builds the output folder paths. |
| `dataset_info.py` | What the SOM code needs to know about the dataset and the CNN, read from the config. |
| `figures.py` | Save every results figure the same way: one PDF, fonts embedded as TrueType. |
| `file_hash.py` | SHA-256 fingerprints of files, used for provenance checks. |
| `frozen_split.py` | Rebuild the project's frozen data split from the saved sample IDs. |
| `heatmaps.py` | Draws neuron-by-neuron heatmaps on the SOM grid. |
| `hotspots.py` | The project's error-hotspot rule (Error Pattern Analysis). |
| `image_transforms.py` | Label-preserving image changes for targeted augmentation (small affine moves only). |
| `logging.py` | Sets up logging to the console and to a log file. |
| `paired_stats.py` | Uncertainty for the arm-against-arm comparison. |
| `protocol_stamp.py` | Freeze the retraining protocol, and refuse the test split until it is frozen. |
| `repo_paths.py` | Resolve paths from configs safely inside a repository root. |
| `retraining_breakdown.py` | Exploratory breakdown of the test results: errors fixed per seed, and error per class. |
| `retraining_cli.py` | Command line shared by the retraining runners (curriculum comparison, targeted augmentation). |
| `retraining_data.py` | Everything a retraining run reads: the frozen split, the hard-image pools and the hotspot mask. |
| `retraining_figures.py` | Confusion-matrix figures: CF1 (before), CF2 (after) and their difference. |
| `retraining_final.py` | The single test pass: evaluate every finished run on the test split, once, after the protocol is frozen. |
| `retraining_metrics.py` | Metrics of one retrained CNN (plain NumPy, so they work for any dataset with N classes). |
| `retraining_plans.py` | What one arm trains on (torch-free, so the arm builders can be tested without PyTorch). |
| `retraining_runs.py` | Run one arm with one seed, save it, and loop over arms and seeds. |
| `retraining_settings.py` | Settings shared by the retraining experiments (curriculum comparison, targeted augmentation). |
| `retraining_stability.py` | Training-stability monitoring: how validation accuracy moves from epoch to epoch. |
| `retraining_summary.py` | Validation-only summary of an experiment: mean and spread over seeds, and arm-against-arm differences. |
| `run_ids.py` | Check run IDs before a step creates its output folder. |
| `run_provenance.py` | Run-ID preflight and SHA-256 provenance helpers shared by the Phase 4 steps. |
| `run_settings.py` | Which SOM the cluster and error pattern analysis uses, and checks that it is the right one. |
| `scores.py` | Per-sample scores computed from the CNN's output that several analyses share. |
| `shared_provenance.py` | Validate Phase 3 frozen input provenance against the shared manifest. |
| `som_cells.py` | Assign embeddings to the neurons (cells) of the selected SOM. |
| `som_data.py` | Data loading and validation utilities for the SOM experiments. |
| `som_figure_setup.py` | Shared setup for the SOM figure scripts of SOM training, cluster analysis and error pattern analysis. |
| `som_io.py` | Loads the embedding splits and the chosen SOM, rebuilds the neuron of every image and writes the assignment tables. |
| `som_neuron_maps.py` | Per-neuron class and error statistics, and the simple grid map used by the cluster analysis and error pattern analysis figure scripts (cluster_figures.py, error_figures.py). |
| `som_plotting.py` | Shared SOM plotting helpers for every week. |
| `som_text_style.py` | Text styling for NNSOM's dense SOM plots (neuron labels on hexagons). |
| `som_types.py` | Type names for annotations of NNSOM objects. |
| `stats.py` | Entropy, the Wilson bound and Spearman correlation. |
