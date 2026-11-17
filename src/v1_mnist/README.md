# MNIST code, tests and documents (`src/v1_mnist/`)

Everything needed to run the MNIST study: the code, its tests and the written protocols and results. `v1_mnist` is the first versioned code folder; the Fashion-MNIST study will get its own folder next to it.

| Folder | What is inside |
|---|---|
| [`component/`](component/README.md) | The code, one folder per project phase and step, plus `pipeline/` (one runner per step and `run_all.py`), `configs/` (every setting) and `utils/` (shared helpers) |
| [`tests/`](tests/README.md) | pytest tests, in the same phase and step folders as `component/` |
| [`docs/`](docs/README.md) | Protocols written before the runs, results, explanations, and `DEVIATIONS.md` (every change to a written rule) |
| [`shellscripts/`](shellscripts/README.md) | Kept for shell scripts; empty for now (the Python runners in `component/pipeline/` are used instead) |

The results of the code are written to `outputs/v1_mnist/`, in the same phase and step folders.

## The four phases

| Phase | Code folder | Steps, in order |
|---|---|---|
| 1: Model Training Baseline | `component/p1_model_training_baseline/` | `data_pipeline`, `cnn_baseline_training` |
| 2: SOM Training & Visualization | `component/p2_som_training_visualization/` | `som_nnsom_training` (with the grid and feature-layer ablations), `cluster_analysis`, `error_pattern_analysis`, `interactive_report` |
| 3: Novelty Detection | `component/p3_novelty_detection/` | `qe_novelty_detection`, `u_matrix_analysis`, `novelty_benchmark`, `held_out_class_novelty` |
| 4: Guided Retraining | `component/p4_guided_retraining/` | `hard_example_mining`, `targeted_augmentation`, `som_reanalysis`, `novelty_after_retraining` |

Run the tests from the code root (the folder that holds `src/` and `outputs/`):

```bash
python -m pytest src/v1_mnist/tests -q
```
