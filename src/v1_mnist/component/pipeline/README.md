# Pipeline: How to Run

Run every command from the project root (on EC2: `~/Capstone/Codes/code_base`).

## Files, in run order

| #   | File                           | What it does                                       | Run it?                              |
| --- | ------------------------------ | -------------------------------------------------- | ------------------------------------ |
| 1   | `prepare_data.py`              | Loads MNIST and makes the train / val / test split | ❌ No, already done                  |
| 2   | `train_baseline.py`            | Trains the LeNet-5 CNN                             | ❌ No, already done                  |
| 3   | `extract_embeddings.py`        | Saves the 84-number embedding of every image       | ❌ No, already done                  |
| 4   | `train_som.py`                 | Trains the 15x15 SOM                               | ✅ Yes                               |
| 5   | `visualize_som.py`             | Makes the static SOM figures                       | ✅ Yes                               |
| 6   | `visualize_som_interactive.py` | Makes the interactive HTML report                  | ✅ Yes                               |
| –   | `runner.py`                    | Runs the stages above in one go                    | ❌ No (it would rerun stages 1 to 3) |

Stages 1 to 3 made the embeddings that all our results use. Running them again would replace those files and change every result.

## Already done, no need to run the following:

```bash
python src/v1_mnist/component/pipeline/prepare_data.py
python src/v1_mnist/component/pipeline/train_baseline.py
python src/v1_mnist/component/pipeline/extract_embeddings.py --checkpoint outputs/v1_mnist/cnn_baseline/checkpoints/lenet5_best.pth
```

## Run the 15x15 v2 SOM

```bash
python src/v1_mnist/component/pipeline/train_som.py --run-id som_15x15_nb11_ep250_s42_v2
python src/v1_mnist/component/pipeline/visualize_som.py
python src/v1_mnist/component/pipeline/visualize_som_interactive.py
```

If `som_15x15_nb11_ep250_s42_v2` is already trained, the first command stops on purpose. Skip it and run only the two visualization commands.

## Where the results go

- Model: `outputs/v1_mnist/som/som_models/som_15x15_nb11_ep250_s42_v2`
- Static figures: `outputs/v1_mnist/som/figures/som_15x15_nb11_ep250_s42_v2/`
- Interactive report: `outputs/v1_mnist/som/som_interactive/interactive_report.html`
