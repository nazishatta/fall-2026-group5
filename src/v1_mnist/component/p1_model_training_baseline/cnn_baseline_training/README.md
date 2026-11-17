# CNN baseline training (`src/v1_mnist/component/p1_model_training_baseline/cnn_baseline_training/`)

Proposal Week 2. Trains the LeNet-5 CNN, tests it once on the held-out test split (99.18% accuracy), and saves the 84-number fc2 embedding of every image. Every later phase works on these embeddings, so this step is run once and its results are frozen.

Runner: `pipeline/p1_model_training_baseline/run_cnn_baseline_training.py` (it refuses to run while its results exist). Results: `outputs/v1_mnist/p1_model_training_baseline/cnn_baseline_training/`. Notes: `docs/p1_model_training_baseline/cnn_baseline_training/cnn_baseline.md`.

## Files

| File | What it is |
|---|---|
| `cnn_plots.py` | Week 2 (Train Baseline CNN) figures: training curves and the test confusion matrix. |
| `evaluation.py` | Test-set accuracy, the classification report (precision, recall, F1 per digit) and the confusion matrix. |
| `extract_embeddings.py` ▶ | Writes the 84-number fc2 embedding, label, prediction and confidence of every image in every split. |
| `extractor.py` | Reads the fc2 layer of the trained LeNet-5 to get the embeddings. |
| `lenet5.py` | The LeNet-5 network. |
| `run_ids.py` | Run IDs for the CNN baseline: where each run's files go, and the overwrite check. |
| `train_baseline.py` ▶ | Trains LeNet-5 and keeps the checkpoint with the best validation accuracy. |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
