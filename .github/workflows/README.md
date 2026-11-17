# GitHub Actions (`.github/workflows/`)

`tests.yml` runs the whole test suite (`pytest src/v1_mnist/tests -q`) on every push and pull request, on Ubuntu with Python 3.13, CPU-only PyTorch and the pinned `requirements.txt`. A pull request should only be merged when this check passes.
