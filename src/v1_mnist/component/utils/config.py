from __future__ import annotations

import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any
import numpy as np
import yaml

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


# Run names from before the folders followed the proposal's phases. They are
# refused (not silently mapped) so that no output folder is named after a week.
OLD_RUN_NAMES = {"week_2": "cnn_baseline", "week_3": "som"}


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge a run config's settings on top of the base settings."""
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def load_config(config_path: str | Path) -> SimpleNamespace:
    """Load configuration YAML and return nested SimpleNamespace."""
    cfg_str = str(config_path)
    with open(cfg_str, "r", encoding="utf-8") as f:
        run_config: dict[str, Any] = yaml.safe_load(f) or {}

    config_dir = os.path.dirname(os.path.abspath(cfg_str))
    base_path = os.path.join(config_dir, "base.yaml")

    if os.path.basename(cfg_str) != "base.yaml" and os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as f:
            base_config: dict[str, Any] = yaml.safe_load(f) or {}
        config_dict = merge_dicts(base_config, run_config)
    else:
        config_dict = run_config

    if "run" in config_dict and "name" in config_dict["run"]:
        raw_run_name = str(config_dict["run"]["name"])
        if raw_run_name in OLD_RUN_NAMES:
            raise ValueError(
                f"run.name '{raw_run_name}' is an old name. "
                f"Use '{OLD_RUN_NAMES[raw_run_name]}' instead."
            )
        # Output folders follow the proposal: <output_root>/p<phase>_<name>/<step folder>/
        if raw_run_name == "cnn_baseline":
            folder_name = os.path.join("p1_model_training_baseline", "cnn_baseline_training")
        elif raw_run_name == "som":
            folder_name = os.path.join("p2_som_training_visualization", "som_nnsom_training")
        else:
            folder_name = raw_run_name

        output_root = config_dict.get("project", {}).get(
            "output_root", "./outputs/v1_mnist"
        )

        if "paths" not in config_dict:
            config_dict["paths"] = {}

        output_dir = os.path.join(output_root, folder_name)

        config_dict["paths"].setdefault("output_dir", output_dir)
        config_dict["paths"].setdefault(
            "checkpoints_dir", os.path.join(output_dir, "checkpoints")
        )
        config_dict["paths"].setdefault(
            "embeddings_dir", os.path.join(output_dir, "embeddings")
        )
        config_dict["paths"].setdefault(
            "figures_dir", os.path.join(output_dir, "figures")
        )
        config_dict["paths"].setdefault(
            "metrics_dir", os.path.join(output_dir, "metrics")
        )
        config_dict["paths"].setdefault(
            "tables_dir", os.path.join(output_dir, "tables")
        )
        config_dict["paths"].setdefault(
            "som_models_dir", os.path.join(output_dir, "som_models")
        )
        config_dict["paths"].setdefault(
            "logs_dir", os.path.join(output_dir, "logs")
        )
        config_dict["paths"].setdefault(
            "analysis_dir", os.path.join(output_dir, "analysis")
        )

        # Step folders that sit next to the run's own folder.
        p1_dir = os.path.join(output_root, "p1_model_training_baseline")
        p2_dir = os.path.join(output_root, "p2_som_training_visualization")
        config_dict["paths"].setdefault(
            "data_pipeline_dir", os.path.join(p1_dir, "data_pipeline")
        )
        config_dict["paths"].setdefault(
            "cluster_dir", os.path.join(p2_dir, "cluster_analysis")
        )
        config_dict["paths"].setdefault(
            "error_dir", os.path.join(p2_dir, "error_pattern_analysis")
        )
        config_dict["paths"].setdefault(
            "interactive_report_dir", os.path.join(p2_dir, "interactive_report")
        )

    def dict_to_namespace(d: Any) -> Any:
        if isinstance(d, dict):
            for k, v in d.items():
                d[k] = dict_to_namespace(v)
            return SimpleNamespace(**d)
        elif isinstance(d, list):
            return [dict_to_namespace(i) for i in d]
        else:
            return d

    return dict_to_namespace(config_dict)


def require_seed(config: Any) -> int:
    """Return ``training.seed`` from the config; stop if the config does not set one.

    There is no default seed in the code, so every run's seed is visible in its config.
    """
    seed = getattr(getattr(config, "training", None), "seed", None)
    if seed is None:
        raise ValueError(
            "The config has no training.seed. Set it in the config (base.yaml sets "
            "the shared value); the code has no default seed."
        )
    return int(seed)


def set_seed(seed: int) -> None:
    """Set deterministic random seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(config: Any) -> Any:
    """Determine compute device for PyTorch model execution."""
    if not HAS_TORCH or torch is None:
        raise RuntimeError(
            "PyTorch is required to get a torch device, but torch is not installed."
        )
    if hasattr(config, "device") and config.device != "auto":
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
