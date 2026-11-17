# Tests (`src/v1_mnist/tests/`)

Unit and regression tests. The folders have the same names as `component/`, so the tests for `component/<phase>/<step>/` are in `tests/<phase>/<step>/`. Tests that need PyTorch or NNSOM are skipped when those are not installed.

Run every test from the code root:

```bash
python -m pytest src/v1_mnist/tests -q
```

`test_repository_rules.py` checks rules for the whole repository: no figures or large files in git, PDF-only figures, no overwrite options, no seeds hard-coded in the code, no module names starting with a digit, and that every Phase 3 result traces to its run and code.

## Folders

| Folder | What is inside |
|---|---|
| [`p1_model_training_baseline/`](p1_model_training_baseline/README.md) | Tests for the Phase 1 code, one folder per step, with the same names as `component/p1_model_training_baseline/`. |
| [`p2_som_training_visualization/`](p2_som_training_visualization/README.md) | Tests for the Phase 2 code, one folder per step, with the same names as `component/p2_som_training_visualization/`. |
| [`p3_novelty_detection/`](p3_novelty_detection/README.md) | Tests for the Phase 3 code, one folder per step, with the same names as `component/p3_novelty_detection/`. |
| [`p4_guided_retraining/`](p4_guided_retraining/README.md) | Tests for the Phase 4 code, one folder per step, with the same names as `component/p4_guided_retraining/`. |
| [`pipeline/`](pipeline/README.md) | Tests for the code in `component/pipeline/`. |
| [`utils/`](utils/README.md) | Tests for the code in `component/utils/`. |

## Files

| File | What it is |
|---|---|
| `conftest.py` | Adds the code root to the import path so the tests can import the project code. |
| `test_repository_rules.py` | Repository rules that .gitignore alone cannot enforce (project rules, Large files / Figures). |
