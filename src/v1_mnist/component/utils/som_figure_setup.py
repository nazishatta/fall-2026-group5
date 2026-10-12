"""Shared setup for the SOM figure scripts of Weeks 3, 4 and 5.

Loads the config, the saved SOM (grid size from its metrics file) and the
frozen embeddings with the train-fitted scaler, makes the pdf/
figure folders, and writes each script's visualization_manifest.json.
The test split is never used.
"""

from __future__ import annotations

from types import SimpleNamespace
import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.v1_mnist.component.utils.som_types import SOMModel

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as mpl_cm
from matplotlib import colormaps

# Compatibility shim for NNSOM + modern Matplotlib.
# NNSOM 1.8.x still calls matplotlib.cm.get_cmap(),
# which is unavailable in newer Matplotlib versions.
if not hasattr(mpl_cm, "get_cmap"):
    def _compat_get_cmap(name: Any = None, lut: Optional[int] = None) -> Any:
        cmap = colormaps.get_cmap(name)
        if lut is not None:
            cmap = cmap.resampled(lut)
        return cmap
    mpl_cm.get_cmap = _compat_get_cmap


from NNSOM.plots import SOMPlots

from src.v1_mnist.component.utils.som_data import load_som_data
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.dataset_info import dataset_info

DEFAULT_CONFIG = "src/v1_mnist/component/configs/som.yaml"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """--config and --model-name, shared by every figure script."""
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Saved SOM filename/run-id. "
            "Defaults to visualization.selected_model."
        ),
    )


def resolve_path(p: str | Path, repo_root: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_config_and_model_name(config_arg: str, model_name_arg: str | None, repo_root: Path) -> tuple[SimpleNamespace, str]:
    """Read the SOM config; the model defaults to visualization.selected_model."""
    config_path = Path(config_arg)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = (repo_root / config_path).resolve()
    config = load_config(str(config_path))
    model_name = model_name_arg or config.visualization.selected_model
    return config, model_name


def make_figure_dirs(*dirs: Path) -> None:
    """Create the pdf/ folder of every figure folder (figures are saved as PDF only)."""
    for sub in dirs:
        (sub.parent / "pdf" / sub.name).mkdir(
            parents=True,
            exist_ok=True,
        )


def load_model_and_grid(
    model_name: str,
    models_dir: Path,
    logs_dir: Path,
) -> tuple[SOMModel, dict]:
    """Load a saved SOM, inferring grid dimensions from its metrics record."""

    metrics_path = (
        logs_dir
        / f"{model_name}_metrics.json"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            "Could not infer SOM grid because the metrics "
            f"file is missing: {metrics_path}"
        )

    metrics = json.loads(
        metrics_path.read_text(
            encoding="utf-8"
        )
    )

    grid_height = int(
        metrics["som"]["grid_height"]
    )

    grid_width = int(
        metrics["som"]["grid_width"]
    )

    model_path = (
        models_dir
        / model_name
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            model_path
        )

    som = SOMPlots(
        dimensions=(
            grid_height,
            grid_width,
        )
    )

    loaded = som.load_pickle(
        model_path.name,
        str(
            model_path.parent
        )
        + os.sep,
    )

    if (
        loaded is not None
        and hasattr(
            loaded,
            "cluster_data",
        )
    ):
        som = loaded

    return som, metrics


@dataclass
class FigureInputs:
    som: Any
    model_metrics: dict
    data: Any


def load_som_and_data(config: SimpleNamespace, model_name: str, repo_root: Path) -> FigureInputs:
    """Load the embeddings and the saved SOM, and reattach the train-fitted scaler."""
    data = load_som_data(
        embeddings_dir=config.paths.embeddings_dir,
        expected_feature_dim=dataset_info(config).feature_dim,
    )
    som, model_metrics = (
        load_model_and_grid(
            model_name=model_name,
            models_dir=resolve_path(config.paths.som_models_dir, repo_root),
            logs_dir=resolve_path(config.paths.logs_dir, repo_root),
        )
    )
    # Reattach the train-fitted scaler explicitly.
    # This is safer than relying on a pickled bound transform method.
    som.norm_func = (
        data.scaler.transform
    )
    return FigureInputs(som=som, model_metrics=model_metrics, data=data)


def class_map_dict(config: SimpleNamespace, train_dict: dict, val_dict: dict) -> tuple[str, dict]:
    """The split named by visualization.class_map_split ('train' or 'val')."""
    class_split_name = str(
        config.visualization.class_map_split
    )
    if class_split_name == "train":
        return class_split_name, train_dict
    if class_split_name == "val":
        return class_split_name, val_dict
    raise ValueError(
        "visualization.class_map_split "
        "must be 'train' or 'val'."
    )


def write_manifest(
    manifest_dir: Path,
    *,
    figures: str,
    model_name: str,
    inputs: FigureInputs,
    generated: list[str],
    failed: dict[str, str],
    repo_root: Path,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write visualization_manifest.json for one week's figures."""
    data = inputs.data
    manifest: dict[str, Any] = {
        "model_name": model_name,
        "figures": figures,
        "grid": [
            int(inputs.model_metrics["som"]["grid_height"]),
            int(inputs.model_metrics["som"]["grid_width"]),
        ],
        "train_shape": list(data.train.features.shape),
        "validation_shape": list(data.val.features.shape),
        "feature_dim": int(data.train.features.shape[1]),
    }
    manifest.update(extra or {})
    manifest["generated_files"] = [
        str(Path(path).relative_to(repo_root))
        if Path(path).is_absolute()
        else path
        for path in generated
    ]
    manifest["failed_or_skipped_due_to_runtime"] = failed
    manifest["test_evaluated"] = False
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path
