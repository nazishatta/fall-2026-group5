"""Data pipeline: environment check and MNIST splits.

Steps:
  1. Environment check: Python, PyTorch, torchvision, NNSOM, scikit-learn,
     NumPy, matplotlib versions and whether a GPU / CuPy is available.
  2. prepare_data: download MNIST, make the stratified 70/15/15 split, save the
     class-distribution charts, sample images and split_stats.json to
     outputs/v1_mnist/p1_model_training_baseline/data_pipeline/.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p1_model_training_baseline/run_data_pipeline.py [--config CONFIG] [--dry-run]

Stops if split_stats.json already exists (results are never overwritten).
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.v1_mnist.component.pipeline.common import (  # noqa: E402
    CNN_CONFIG, P1, Step, log, base_parser, check_not_done, load, resolve, run_steps,
)

REQUIRED = [  # (import name, package name)
    ("torch", "torch"), ("torchvision", "torchvision"), ("NNSOM", "NNSOM"),
    ("sklearn", "scikit-learn"), ("numpy", "numpy"), ("matplotlib", "matplotlib"), ("yaml", "PyYAML"),
]


def environment_check() -> None:
    """Print library versions and GPU availability; stop if a required library is missing."""
    log.info(f"    Python {platform.python_version()} ({sys.executable})")
    missing = []
    for module, package in REQUIRED:
        try:
            import_module(module)
            try:
                v = version(package)
            except PackageNotFoundError:
                v = "installed"
            log.info(f"    {package:<13} {v}")
        except ImportError:
            log.info(f"    {package:<13} MISSING")
            missing.append(package)
    try:
        import torch
        gpu = torch.cuda.is_available()
        log.info(f"    CUDA GPU      {'yes: ' + torch.cuda.get_device_name(0) if gpu else 'no (CPU only)'}")
    except ImportError:
        pass
    try:
        import_module("cupy")
        log.info("    CuPy          available (NNSOM GPU backend)")
    except ImportError:
        log.info("    CuPy          not installed (NNSOM runs on CPU/NumPy)")
    if missing:
        log.error(f"\nStopped: install the missing packages first: {', '.join(missing)}")
        sys.exit(1)


def main() -> None:
    parser = base_parser("Data pipeline: environment check and MNIST splits.", CNN_CONFIG)
    args = parser.parse_args()

    config = load(args.config)
    data_pipeline_dir = resolve(config.paths.data_pipeline_dir)
    check_not_done([data_pipeline_dir / "metrics" / "split_stats.json"], args.dry_run)

    steps = [
        Step("Environment check", func=environment_check),
        Step("Prepare MNIST: split, class distributions, sample images",
             P1 / "data_pipeline" / "prepare_data.py", ["--config", args.config]),
    ]
    run_steps("Data pipeline: environment check and MNIST splits", steps, args.dry_run)


if __name__ == "__main__":
    main()
