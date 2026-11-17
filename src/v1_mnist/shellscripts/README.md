# Shell scripts (`src/v1_mnist/shellscripts/`)

This folder is kept for shell scripts that run several steps in a row. It is empty for now: the old `run_cnn_baseline.sh` was removed when the code was arranged by phase (commit `48d2ca6`), because the Python runners in `component/pipeline/` now do the same job on every system (Linux, macOS, EC2 and Windows).

To run Phase 1 (data split, then LeNet-5 training and embedding extraction), from the code root:

```bash
python src/v1_mnist/component/pipeline/p1_model_training_baseline/run_data_pipeline.py
python src/v1_mnist/component/pipeline/p1_model_training_baseline/run_cnn_baseline_training.py
```

Both runners stop if their results already exist; results are never overwritten. A new CNN run needs a new `run.id` in `configs/cnn_baseline.yaml`. To run every step in order, use `component/pipeline/run_all.py` (see `component/pipeline/README.md`).
