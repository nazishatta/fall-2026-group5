# Phase 4 tests (`src/v1_mnist/tests/p4_guided_retraining/`)

Tests for the Phase 4 code, one folder per step, with the same names as `component/p4_guided_retraining/`. `synthetic_data.py` is a tiny made-up dataset the Phase 4 tests share. Run them with `python -m pytest src/v1_mnist/tests/p4_guided_retraining -q`.

## Folders

| Folder | What is inside |
|---|---|
| [`hard_example_mining/`](hard_example_mining/README.md) | Tests for the code in `component/p4_guided_retraining/hard_example_mining/`. |
| [`targeted_augmentation/`](targeted_augmentation/README.md) | Tests for the code in `component/p4_guided_retraining/targeted_augmentation/`. |
| [`som_reanalysis/`](som_reanalysis/README.md) | Tests for the code in `component/p4_guided_retraining/som_reanalysis/`. |
| [`novelty_after_retraining/`](novelty_after_retraining/README.md) | Tests for the code in `component/p4_guided_retraining/novelty_after_retraining/`. |

## Files

| File | What it is |
|---|---|
| `synthetic_data.py` | A tiny made-up RetrainingData for the Phase 4 tests (no MNIST files, no PyTorch). |
| `test_dataset_settings.py` | Phase 4 builds its network (channels, image size) and finds its protocol from the dataset's config; MNIST by default. |
| `test_protocol_stamps.py` | Both datasets' stamped Phase 4 protocols and `retraining.yaml` files still match their stamps (results live in separate files). |
