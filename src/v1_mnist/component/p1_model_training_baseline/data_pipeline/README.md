# Data pipeline (`src/v1_mnist/component/p1_model_training_baseline/data_pipeline/`)

Proposal step: Environment Setup & Data Pipeline. Pools the 70,000 MNIST images (the 60,000 training and 10,000 test images), splits them 70/15/15 by digit with seed 42 (49,000 train / 10,500 validation / 10,500 test), and draws the class-distribution charts and a grid of sample images.

Runner: `pipeline/p1_model_training_baseline/run_data_pipeline.py`. Results: `outputs/v1_mnist/p1_model_training_baseline/data_pipeline/`. Tests: `tests/p1_model_training_baseline/data_pipeline/`.

## Files

| File | What it is |
|---|---|
| `data_plots.py` | Environment Setup & Data Pipeline figures: class distributions and sample images. |
| `image_dataset.py` | Loads the dataset named in the config (through `utils/image_datasets.py`) and makes the stratified 70/15/15 train/validation/test split. |
| `prepare_data.py` ▶ | Builds the split, the class-distribution charts and the sample-image grid. |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
