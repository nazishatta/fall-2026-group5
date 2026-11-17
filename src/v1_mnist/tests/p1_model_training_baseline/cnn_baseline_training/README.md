# Tests: cnn baseline training (`src/v1_mnist/tests/p1_model_training_baseline/cnn_baseline_training/`)

Tests for the code in `component/p1_model_training_baseline/cnn_baseline_training/`. Run them with `python -m pytest src/v1_mnist/tests/p1_model_training_baseline/cnn_baseline_training -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_accuracy_interval.py` | CNN test accuracy is reported with a 95% Wilson interval (project rule: report uncertainty). |
| `test_cnn_run_id.py` | CNN run IDs: file names carry the run ID, and existing runs are never overwritten. |
| `test_feature_extractor.py` | The fc2 extractor returns finite 84-number embeddings and correct metadata. |
| `test_lenet5.py` | LeNet-5 output shape and the fc2 feature layer. |
