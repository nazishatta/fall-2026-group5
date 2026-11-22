# Tests: targeted augmentation (`src/v1_mnist/tests/p4_guided_retraining/targeted_augmentation/`)

Tests for the code in `component/p4_guided_retraining/targeted_augmentation/`. Run them with `python -m pytest src/v1_mnist/tests/p4_guided_retraining/targeted_augmentation -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_augmentation_arms.py` | Targeted augmentation: the altered copies and the arms built from them. |
| `test_augmentation_check.py` | The transform check stops the train stage until it has passed, when the config sets a limit (Fashion-MNIST); without one (MNIST) nothing changes. |
