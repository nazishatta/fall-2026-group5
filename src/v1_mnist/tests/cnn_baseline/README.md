# CNN Baseline Tests

This folder contains lightweight tests for the CNN baseline components.

Tests cover:

- MNIST split and image properties
- LeNet-5 model structure
- `fc2` feature extraction

## Setup

From the project root:

```bash
python -m pip install -r requirements.txt
```

## Run All CNN Baseline Tests

On local machines, EC2, or other instances:

```bash
python -m pytest src/v1_mnist/tests/cnn_baseline -v
```

## Run One Test File

Example:

```bash
python -m pytest src/v1_mnist/tests/cnn_baseline/test_lenet5.py -v
```

Run tests from the project root so imports and relative paths resolve consistently.
