# Shell Scripts

This folder contains scripts for reproducing project workflows.

## CNN Baseline

`run_cnn_baseline.sh` runs the Week 1 and Week 2 runners (`pipeline/p1_model_training_baseline/`):

1. Environment check and MNIST preparation with a 70/15/15 split
2. LeNet-5 training and evaluation
3. 84-dimensional `fc2` embedding extraction

Both runners stop if their results already exist; results are never overwritten. A new CNN run needs a new `run.id` in `configs/cnn_baseline.yaml` (and a new `paths.embeddings_dir`).

Outputs are saved to:

```text
outputs/v1_mnist/p1_model_training_baseline/data_pipeline/          (Week 1)
outputs/v1_mnist/p1_model_training_baseline/cnn_baseline_training/  (Week 2)
```

## Setup

From the project root, install the required packages:

```bash
python -m pip install -r requirements.txt
```

Activate your preferred Python environment first if you use one.

## Run on EC2 / Linux / macOS

From the project root:

```bash
bash src/v1_mnist/shellscripts/run_cnn_baseline.sh
```

Optional:

```bash
chmod +x src/v1_mnist/shellscripts/run_cnn_baseline.sh
./src/v1_mnist/shellscripts/run_cnn_baseline.sh
```

## Run on a Local Windows Machine

The `.sh` script can be run with Git Bash or WSL:

```bash
bash src/v1_mnist/shellscripts/run_cnn_baseline.sh
```

Or run the two week runners directly from PowerShell/Command Prompt if Bash is not available:

```bash
python src/v1_mnist/component/pipeline/p1_model_training_baseline/run_data_pipeline.py
python src/v1_mnist/component/pipeline/p1_model_training_baseline/run_cnn_baseline_training.py
```

## Before Running

Make sure:

- Python is installed
- packages in `requirements.txt` are installed
- the MNIST data path in the project configuration is correct
