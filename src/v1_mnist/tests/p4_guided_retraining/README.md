# Phase 4: Guided Retraining (Weeks 9–11) — Tests

> `p4` in the folder name means **Phase 4**.

`hard_example_mining/` tests the difficulty rules, pools, run-ID preflight, fixed pool settings and the curriculum. `targeted_augmentation/` tests the altered copies and the augmented arms. `som_reanalysis/` is empty. `synthetic_data.py` is a tiny made-up dataset the tests share. Tests of the code shared by both steps are in `tests/utils/` (`test_retraining_support.py`; `test_cnn_retraining.py` needs PyTorch).

```
p4_guided_retraining/
├── hard_example_mining/           # hard-example pool, curriculum schedule
├── targeted_augmentation/         # augmentation only in hotspot neurons
└── som_reanalysis/                # old vs. new SOM comparison
```
