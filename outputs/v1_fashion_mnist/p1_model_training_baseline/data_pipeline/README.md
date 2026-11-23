# `data_pipeline` results (Fashion-MNIST)

Phase 1, data pipeline (code: `src/v1_mnist/component/p1_model_training_baseline/data_pipeline/`, shared with MNIST). The 70,000 Fashion-MNIST images are split 70/15/15, stratified by class, with seed 42.

| Path | What is inside | Where it is kept |
|---|---|---|
| `metrics/split_stats.json` | Number of images of each class in the train, validation and test splits (49,000 / 10,500 / 10,500; 4,900 / 1,050 / 1,050 per class) | git |
| `figures/` (+ `figures.dvc`) | Class-distribution charts for each split and a grid of sample images (PDF) | DVC |
