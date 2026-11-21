# Phase 3 tests (`src/v1_mnist/tests/p3_novelty_detection/`)

Tests for the Phase 3 code, one folder per step, with the same names as `component/p3_novelty_detection/`. Run them with `python -m pytest src/v1_mnist/tests/p3_novelty_detection -q`.

## Folders

| Folder | What is inside |
|---|---|
| [`qe_novelty_detection/`](qe_novelty_detection/README.md) | Tests for the code in `component/p3_novelty_detection/qe_novelty_detection/`. |
| [`u_matrix_analysis/`](u_matrix_analysis/README.md) | Tests for the code in `component/p3_novelty_detection/u_matrix_analysis/`. |
| [`novelty_benchmark/`](novelty_benchmark/README.md) | Tests for the code in `component/p3_novelty_detection/novelty_benchmark/`. |
| [`held_out_class_novelty/`](held_out_class_novelty/README.md) | Tests for the code in `component/p3_novelty_detection/held_out_class_novelty/`. |

Files directly in this folder:

| File | What it is |
|---|---|
| `test_dataset_names.py` | Phase 3 result names follow the in-distribution dataset; MNIST keeps its historical names. |
