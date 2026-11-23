# Tests, p4_guided_retraining (Fashion-MNIST)

Tests for the Fashion-MNIST p4_guided_retraining settings. The code is shared with MNIST and tested in
`src/v1_mnist/tests/p4_guided_retraining/`.

| File | What it checks |
|---|---|
| `test_phase4_configs.py` | The four Phase 4 configs copy MNIST's settings except the dataset's paths and the protocol's own values (macro-F1 guard, protocol and stamp files, transform-check limit, and the augmentation ranges: MNIST's halved by the rule in `docs/DEVIATIONS.md`); the guard is empty or equals its rule (2 x the `standard` arm's validation macro-F1 spread, the larger of the two experiments). |
