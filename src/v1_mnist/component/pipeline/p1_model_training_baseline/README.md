# Phase 1 runners (`src/v1_mnist/component/pipeline/p1_model_training_baseline/`)

One runner per Phase 1 step, named after the step folder it runs (`run_<step>.py` runs `component/p1_model_training_baseline/<step>/`). Run from the code root (the folder that holds `src/` and `outputs/`). Each runner reads its settings from `component/configs/`, checks that its run ID is new, and never overwrites a finished run. See `component/pipeline/README.md` for which runners `run_all.py` calls and which are run on their own.

## Files

| File | What it is |
|---|---|
| `run_cnn_baseline_training.py` ▶ | CNN baseline training: train LeNet-5 and extract embeddings. |
| `run_data_pipeline.py` ▶ | Data pipeline: environment check and MNIST splits. |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
