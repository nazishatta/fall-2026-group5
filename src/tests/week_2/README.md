# Tests

This folder contains lightweight tests for the main project components.

Week 2 tests cover:

- MNIST split and image properties
- LeNet-5 model structure
- `fc2` feature extraction

## Setup

From the project root:

```bash
python -m pip install -r requirements.txt
```

## Run All Week 2 Tests

On local machines, EC2, or other instances:

```bash
python -m pytest src/tests/week_2 -v
```

## Run One Test File

Example:

```bash
python -m pytest src/tests/week_2/test_lenet5.py -v
```

Run tests from the project root so imports and relative paths resolve consistently.
