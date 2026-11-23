# Tests: targeted augmentation (`src/v1_mnist/tests/p4_guided_retraining/targeted_augmentation/`)

Tests for the code in `component/p4_guided_retraining/targeted_augmentation/`. Run them with `python -m pytest src/v1_mnist/tests/p4_guided_retraining/targeted_augmentation -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_augmentation_arms.py` | Targeted augmentation: the altered copies and the arms built from them. |
| `test_augmentation_check.py` | The transform check stops the train stage until it has passed, when the config sets a limit (Fashion-MNIST); without one (MNIST) nothing changes. |
| `test_transform_rule.py` | The halving rule in code: halving MNIST's ranges twice gives Fashion-MNIST's; fed the drops recorded in Fashion-MNIST's r1–r3 checks, it stops at r3 as was done by hand; the check stage records every attempt and training uses the accepted ranges; the current configs do not turn it on. |
