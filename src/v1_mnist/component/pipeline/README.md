# Pipeline: How to Run

Run everything from the project root (`week_6_codes_local`). The stages must run **in this order**, because each one uses files made by the one before it.

| # | Script | What it does |
|---|--------|--------------|
| 1 | `prepare_data.py` | Loads MNIST and makes the train / val / test split |
| 2 | `train_baseline.py` | Trains the LeNet-5 model |
| 3 | `extract_embeddings.py` | Saves the 84-number embedding of every image |
| 4 | `train_som.py` | Trains the 15x15 SOM |
| 5 | `visualize_som.py` | Makes the static SOM figures |
| 6 | `visualize_som_interactive.py` | Makes the interactive HTML report |

## Option A: run all 6 stages with one command

```bash
python src/v1_mnist/component/pipeline/runner.py --stage all --run-id som_15x15_seed42_final
```

Add `--overwrite` if that run ID already exists.

## Option B: run one stage at a time

```bash
python src/v1_mnist/component/pipeline/prepare_data.py
python src/v1_mnist/component/pipeline/train_baseline.py
python src/v1_mnist/component/pipeline/extract_embeddings.py --checkpoint outputs/v1_mnist/cnn_baseline/checkpoints/lenet5_best.pth
python src/v1_mnist/component/pipeline/train_som.py --run-id som_15x15_seed42_final
python src/v1_mnist/component/pipeline/visualize_som.py
python src/v1_mnist/component/pipeline/visualize_som_interactive.py
```

To run just one stage with the runner, use for example `--stage train_som`.

## Where the results go

- Static figures: `outputs/v1_mnist/som/figures/`
- Interactive report: `outputs/v1_mnist/som/som_interactive/interactive_report.html` (double-click to open in a browser)

## Note

Stage 3 saves the embeddings in `outputs/v1_mnist/cnn_baseline/embeddings/`, but `som.yaml` reads them from `embeddings/mnist/`. If stage 4 says "Missing Week 2 embedding artifacts", copy the `.npy` files into an `embeddings/mnist/` folder (or change `embeddings_dir` in `som.yaml`).
