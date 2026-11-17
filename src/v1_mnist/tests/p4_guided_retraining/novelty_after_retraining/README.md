# Tests: novelty after retraining (`src/v1_mnist/tests/p4_guided_retraining/novelty_after_retraining/`)

Tests for the code in `component/p4_guided_retraining/novelty_after_retraining/`. Run them with `python -m pytest src/v1_mnist/tests/p4_guided_retraining/novelty_after_retraining -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_comparison.py` | Statements N1 to N4 and the config of the novelty-after-retraining step. |
| `test_scoring.py` | Scores and per-cell metrics of the novelty-after-retraining step (synthetic data, no models). |
