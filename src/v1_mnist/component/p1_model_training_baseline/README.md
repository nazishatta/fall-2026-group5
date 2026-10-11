# Phase 1: Model Training Baseline

Trains the LeNet-5 CNN on MNIST and saves the embeddings every SOM result is built on. Both steps are done and frozen.

```
p1_model_training_baseline/
├── data_pipeline/                 1. proposal week 1
│   ├── prepare_data.py            ▶ entry point: split, class counts, figures
│   ├── mnist_dataset.py           MNIST loading and the stratified 70/15/15 split
│   └── data_plots.py              class-distribution charts, sample images
└── cnn_baseline_training/         2. proposal week 2
    ├── train_baseline.py          ▶ entry point 1: train LeNet-5, test metrics
    ├── extract_embeddings.py      ▶ entry point 2: save the 84-D fc2 embeddings
    ├── lenet5.py                  the model
    ├── extractor.py               reads the fc2 layer; saves/loads embeddings
    ├── evaluation.py              accuracy and per-class results
    ├── cnn_plots.py               training curves, confusion matrix
    └── run_ids.py                 run IDs and the never-overwrite check
```

**Order:** `data_pipeline` → `cnn_baseline_training` (train, then extract). ▶ = script you run.
Runners: `pipeline/p1_model_training_baseline/run_data_pipeline.py`, then `run_cnn_baseline_training.py`.
Results: `outputs/v1_mnist/p1_model_training_baseline/`. Each step folder also has an `*_EXPLAINED.md`.
