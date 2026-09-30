# Shell Scripts

This folder contains scripts for reproducing project workflows.

## CNN Baseline

`run_cnn_baseline.sh` runs:

1. MNIST preparation with a 70/15/15 split
2. LeNet-5 training and evaluation
3. 84-dimensional `fc2` embedding extraction

Outputs are saved to:

```text
outputs/v1_mnist/cnn_baseline/
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

Or run the three Python pipeline commands directly from PowerShell/Command Prompt if Bash is not available.

## Before Running

Make sure:

- Python is installed
- packages in `requirements.txt` are installed
- the MNIST data path in the project configuration is correct
