# `cnn_baseline_training` results (Fashion-MNIST)

Phase 1, the LeNet-5 baseline, trained on Fashion-MNIST with the MNIST recipe (code: `src/v1_mnist/component/p1_model_training_baseline/cnn_baseline_training/`; config: `src/v1_fashion_mnist/component/configs/cnn_baseline.yaml`). Run ID `lenet5_fashion_mnist_s42` (seed 42). Every SOM in Phases 2 and 3 is built from this network's `fc2` embeddings.

| Path | What is inside | Where it is kept |
|---|---|---|
| `checkpoints/lenet5_fashion_mnist_s42.pth` | The trained network | DVC (`checkpoints.dvc`) |
| `metrics/lenet5_fashion_mnist_s42_test_evaluation.json` | Test accuracy, per-class report and confusion matrix | git |
| `metrics/lenet5_fashion_mnist_s42_run.json` | Run record: seed, config and checkpoint fingerprints, library versions | git |
| `embeddings/` | The `fc2` embeddings (84 numbers per image) of every split, with `embedding_manifest.json` | DVC (`embeddings.dvc`) |
| `figures/` | Training curves and the test confusion matrix (PDF) | DVC (`figures.dvc`) |
| `logs/` | Training, embedding extraction and data-pipeline logs | DVC (`logs.dvc`) |

Main result: test accuracy 90.9% (macro F1 0.909); the weakest class is Shirt (F1 0.746). The test split was used only for this Phase 1 evaluation.
