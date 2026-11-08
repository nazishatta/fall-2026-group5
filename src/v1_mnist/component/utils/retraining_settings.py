"""Settings shared by the retraining experiments (curriculum comparison, targeted augmentation).

Everything comes from retraining.yaml, the Phase 1 CNN config (recipe and MNIST folder) and the
SOM config, so no number is hard-coded in the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.settings import (
    DEFAULT_CONFIG,
    load_pool_settings,
    resolve_path,
)
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.run_settings import AnalysisSettings

ARM_KINDS = ("standard", "curriculum", "augmentation")
POOL_KINDS = ("som", "msp", "random", "hotspot", "random_hotspot")


@dataclass(frozen=True)
class TrainSettings:
    """The Phase 1 training recipe."""

    epochs: int
    lr: float
    weight_decay: float
    step_size: int
    gamma: float
    batch_size: int
    num_classes: int


@dataclass(frozen=True)
class ArmSpec:
    """One way of training. ``pool`` and ``schedule`` are empty for the standard arm."""

    name: str
    kind: str
    pool: str = ""
    schedule: str = ""


@dataclass(frozen=True)
class ExperimentSettings:
    """One experiment (a section of retraining.yaml) and the inputs it reads."""

    section: str
    config_path: Path
    run_id: str
    output_dir: Path
    seeds: tuple[int, ...]
    device: str
    train: TrainSettings
    arms: tuple[ArmSpec, ...]
    options: dict[str, Any]
    data_root: Path
    cnn_checkpoint: Path
    embeddings_dir: Path
    pool_dir: Path
    bootstrap_resamples: int
    bootstrap_seed: int
    macro_f1_guard: float | None
    damage_tolerance: float
    protocol_path: Path
    stamp_path: Path
    som_config: str
    analysis: AnalysisSettings

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def predictions_dir(self) -> Path:
        return self.output_dir / "predictions"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    def metrics_path(self, arm: str, seed: int) -> Path:
        return self.metrics_dir / f"{arm}_seed{seed}.json"

    def checkpoint_path(self, arm: str, seed: int) -> Path:
        return self.checkpoints_dir / f"{arm}_seed{seed}.pth"

    def prediction_path(self, arm: str, seed: int, split: str) -> Path:
        return self.predictions_dir / f"{arm}_seed{seed}_{split}.npz"


def _arm(entry: dict[str, Any]) -> ArmSpec:
    arm = ArmSpec(
        name=str(entry["name"]),
        kind=str(entry["kind"]),
        pool=str(entry.get("pool", "")),
        schedule=str(entry.get("schedule", "")),
    )
    if arm.kind not in ARM_KINDS:
        raise ValueError(f"Arm {arm.name}: kind must be one of {ARM_KINDS}.")
    if arm.kind != "standard" and arm.pool not in POOL_KINDS:
        raise ValueError(f"Arm {arm.name}: pool must be one of {POOL_KINDS}.")
    return arm


def load_experiment_settings(
    section: str, config_path: str | Path | None, repo_root: Path
) -> ExperimentSettings:
    """Read one experiment section ('curriculum' or 'augmentation') of retraining.yaml."""
    cfg_path = resolve_path(repo_root, config_path or DEFAULT_CONFIG)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    pool_settings, analysis = load_pool_settings(cfg_path, repo_root)
    cnn = load_config(str(resolve_path(repo_root, raw["inputs_cnn"]["cnn_config"])))
    block = raw[section]

    revision = block["run"].get("revision") or ""
    run_id = f"{block['run']['name']}_{analysis.selected_model}" + (f"_{revision}" if revision else "")
    guard = raw["evaluation"].get("macro_f1_guard")
    options = {k: v for k, v in block.items() if k not in ("run", "paths", "arms")}
    return ExperimentSettings(
        section=section,
        config_path=cfg_path,
        run_id=run_id,
        output_dir=resolve_path(repo_root, block["paths"]["output_root"]) / run_id,
        seeds=tuple(int(s) for s in raw["training"]["seeds"]),
        device=str(raw["training"].get("device", "auto")),
        train=TrainSettings(
            epochs=int(cnn.training.epochs),
            lr=float(cnn.training.lr),
            weight_decay=float(cnn.training.weight_decay),
            step_size=int(cnn.training.step_size),
            gamma=float(cnn.training.gamma),
            batch_size=int(cnn.dataloader.batch_size),
            num_classes=int(cnn.dataset.num_classes),
        ),
        arms=tuple(_arm(e) for e in block["arms"]),
        options=options,
        data_root=resolve_path(repo_root, cnn.paths.data_root),
        cnn_checkpoint=resolve_path(repo_root, cnn.paths.checkpoints_dir) / f"{cnn.run.id}.pth",
        embeddings_dir=analysis.embeddings_dir,
        pool_dir=pool_settings.output_dir,
        bootstrap_resamples=int(raw["evaluation"]["bootstrap"]["resamples"]),
        bootstrap_seed=int(raw["evaluation"]["bootstrap"]["seed"]),
        macro_f1_guard=None if guard is None else float(guard),
        damage_tolerance=float(raw["evaluation"]["non_hotspot_damage_tolerance"]),
        protocol_path=resolve_path(repo_root, raw["protocol"]["document"]),
        stamp_path=resolve_path(repo_root, raw["protocol"]["stamp"]),
        som_config=pool_settings.som_config,
        analysis=analysis,
    )
