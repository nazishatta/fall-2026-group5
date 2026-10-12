# Shared code (`component/utils/`)

Helpers used by more than one step of the project. Nothing here runs on its own: every file is imported by the step scripts in `p1_model_training_baseline/`, `p2_som_training_visualization/` and `pipeline/`, or by another file in this folder.

The trees below follow the code's own layout: each **phase folder**, then its **step folders** (one per proposal week), then each script, and under the script the **utils files it calls** with the functions it imports. Paths are relative to `src/v1_mnist/component/`.

## The files

| File                  | What it does                                                                                                                                                                             |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`           | Loads a YAML config with `base.yaml` merged in (`load_config`); sets seeds (`set_seed`, `require_seed`); picks the device (`get_device`). Refuses the old run names `week_2` / `week_3`. |
| `dataset_info.py`     | What the SOM code needs to know about the dataset and the CNN (classes, feature size, display names), read from the config.                                                              |
| `figures.py`          | Saves every results figure the same way: one PDF with embedded fonts (`save_pdf`).                                                                                                       |
| `heatmaps.py`         | Reads per-neuron tables and draws the cluster and error heatmaps.                                                                                                                        |
| `hotspots.py`         | The project's one error-hotspot rule: top max(5, 5% of neurons) by Wilson lower bound.                                                                                                   |
| `logging.py`          | Loggers: `setup_logger` (console and log file), `get_logger`, `get_console_logger` (the runners' plain console output).                                                                  |
| `run_settings.py`     | Which SOM the cluster and error-pattern analyses use, where their outputs go, and checks that the model and inputs are the right ones (SHA-256).                                         |
| `som_data.py`         | Loads and checks the CNN embeddings from the configured folder only; fits the scaler on the training split only.                                                                         |
| `som_figure_setup.py` | Shared start-up for the three SOM figure scripts: arguments, config, model, data, figure folders, figure manifest.                                                                       |
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
