"""Which SOM the RQ1/RQ2 analysis uses, and checks that it is the right one.

Everything comes from a SOM config (som.yaml or a robustness config):
grid size, the selected model's run ID, and the embeddings folder.
The model is accepted only if its SHA-256 matches the hash recorded
when it was trained (outputs/v1_mnist/som/logs/run_manifest.csv, written
by train_som.py) and the hash stored in its metrics file.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from src.v1_mnist.component.analysis.io import sha256_file
from src.v1_mnist.component.utils.config import load_config

DEFAULT_CONFIG = "src/v1_mnist/component/configs/som.yaml"


@dataclass(frozen=True)
class RQSettings:
    """Everything the RQ analysis needs to know about the selected SOM."""

    config_path: Path
    grid_height: int
    grid_width: int
    selected_model: str
    model_path: Path
    metrics_path: Path
    manifest_path: Path
    embeddings_dir: Path
    expected_model_sha256: str
    expected_inputs: dict[str, str]

    @property
    def num_neurons(self) -> int:
        return self.grid_height * self.grid_width

    @property
    def analysis_run_id(self) -> str:
        return f"{self.selected_model}_rq1_rq2"


def _resolve(repo_root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (repo_root / p).resolve()


def expected_hash_from_manifest(manifest_path: Path, run_id: str) -> str:
    """Model SHA-256 recorded for run_id; exactly one manifest row must match."""
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Run manifest not found: {manifest_path}. Train the model with "
            "train_som.py (it writes the manifest) before running the analysis."
        )
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("run_id") == run_id]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one manifest row for run_id '{run_id}' in "
            f"{manifest_path}, found {len(rows)}."
        )
    return rows[0]["model_sha256"]


def load_rq_settings(config_path: str | Path | None, repo_root: Path) -> RQSettings:
    """Read the SOM config and locate the selected model and its records."""
    cfg_path = _resolve(repo_root, config_path or DEFAULT_CONFIG)
    config = load_config(str(cfg_path))

    selected = str(config.visualization.selected_model)
    model_path = _resolve(repo_root, config.paths.som_models_dir) / selected
    logs_dir = _resolve(repo_root, config.paths.logs_dir)
    metrics_path = logs_dir / f"{selected}_metrics.json"
    manifest_path = logs_dir / "run_manifest.csv"

    embeddings_dir = _resolve(repo_root, config.paths.embeddings_dir)
    if not (embeddings_dir / "train_features.npy").is_file():
        raise FileNotFoundError(f"Embeddings not found in the configured folder: {embeddings_dir}")

    expected_inputs = {}
    exp = getattr(config, "expected_inputs", None)
    if exp is not None:
        expected_inputs = {k: str(v) for k, v in vars(exp).items()}

    return RQSettings(
        config_path=cfg_path,
        grid_height=int(config.som.grid_height),
        grid_width=int(config.som.grid_width),
        selected_model=selected,
        model_path=model_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        embeddings_dir=embeddings_dir,
        expected_model_sha256=expected_hash_from_manifest(manifest_path, selected),
        expected_inputs=expected_inputs,
    )


def verify_selected_model(settings: RQSettings) -> str:
    """Refuse to analyse a model whose bytes differ from the recorded ones."""
    if not settings.model_path.is_file():
        raise FileNotFoundError(f"SOM model file not found: {settings.model_path}")
    observed = sha256_file(settings.model_path)
    if observed != settings.expected_model_sha256:
        raise RuntimeError(
            f"Model integrity check failed for {settings.model_path}.\n"
            f"Expected (run_manifest.csv): {settings.expected_model_sha256}\n"
            f"Observed:                    {observed}"
        )
    metrics = json.loads(settings.metrics_path.read_text(encoding="utf-8"))
    recorded = (metrics.get("model") or {}).get("sha256")
    if recorded is not None and recorded != observed:
        raise RuntimeError(
            f"Model hash in {settings.metrics_path.name} ({recorded}) does not "
            f"match the model file ({observed})."
        )
    return observed


def verify_inputs(settings: RQSettings, observed: dict[str, str]) -> None:
    """Check embeddings against the frozen hashes listed in the config."""
    for key, file_key in (("train_features_sha256", "train_features"),
                          ("val_features_sha256", "val_features")):
        expected = settings.expected_inputs.get(key)
        got = observed.get(file_key)
        if expected and got and expected != got:
            raise RuntimeError(
                f"Embedding check failed for {file_key}: expected {expected}, got {got}."
            )
