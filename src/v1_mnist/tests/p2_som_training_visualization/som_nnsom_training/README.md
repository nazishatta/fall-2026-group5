# Tests: som nnsom training (`src/v1_mnist/tests/p2_som_training_visualization/som_nnsom_training/`)

Tests for the code in `component/p2_som_training_visualization/som_nnsom_training/`. Run them with `python -m pytest src/v1_mnist/tests/p2_som_training_visualization/som_nnsom_training -q` from the code root.

## Folders

| Folder | What is inside |
|---|---|
| [`grid_comparison/`](grid_comparison/README.md) | Tests for the code in `component/p2_som_training_visualization/som_nnsom_training/grid_comparison/`. |
| [`layer_ablation/`](layer_ablation/README.md) | Tests for the code in `component/p2_som_training_visualization/som_nnsom_training/layer_ablation/`. |

## Files

| File | What it is |
|---|---|
| `test_dataset_info.py` | Dataset and model facts come from the config, not from the code (utils/dataset_info.py). |
| `test_embeddings_folder.py` | SOM code reads embeddings only from the configured folder, never a fallback copy. |
| `test_memory_safe_init.py` | Regression tests for the NNSOM memory-safe initialization workaround. |
| `test_model_save_path.py` | Regression test for NNSOM model save-path handling. |
| `test_optuna_selection.py` | Tests for the Optuna tuning rules and config (no SOM training). |
| `test_run_id_preflight.py` | Regression tests for v1_mnist scientific run output isolation. |
| `test_som_figure_formats.py` | SOM Setup with NNSOM: figures are saved as one PDF each, with embedded TrueType fonts (pdf.fonttype 42), and dense plots rasterize their data layer only. |
| `test_trainer_vs_nnsom.py` | Check the history trainer against NNSOM's native train(). |
