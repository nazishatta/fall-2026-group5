# Phase 1: Model Training Baseline (`src/v1_mnist/component/p1_model_training_baseline/`)

Prepares the data and trains the CNN whose embeddings every later phase studies. **Order:** `data_pipeline` → `cnn_baseline_training`. Runners in `pipeline/p1_model_training_baseline/`, tests in `tests/p1_model_training_baseline/`, results in `outputs/v1_mnist/p1_model_training_baseline/`.

## Folders

| Folder | What is inside |
|---|---|
| [`data_pipeline/`](data_pipeline/README.md) | Proposal step Environment Setup & Data Pipeline: pools the 70,000 MNIST images (the 60,000 training and 10,000 test images), splits them 70/15/15 by digit with seed 42 (49,000 train / 10,500 validation / 10,500 test), and draws the class-distribution charts and a grid of sample images. |
| [`cnn_baseline_training/`](cnn_baseline_training/README.md) | Proposal step Train Baseline CNN: trains the LeNet-5 CNN, tests it once on the held-out test split (99.18% accuracy), and saves the 84-number fc2 embedding of every image. |
