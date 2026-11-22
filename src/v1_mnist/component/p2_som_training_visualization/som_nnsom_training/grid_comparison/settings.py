"""Settings of the grid comparison: grid_comparison.yaml plus the configs it names.

Rules: docs/p2_som_training_visualization/som_nnsom_training/PREREGISTRATION_grid_comparison.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.settings import resolve_path
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.run_ids import check_run_id

DEFAULT_CONFIG = "src/v1_mnist/component/configs/grid_comparison.yaml"
EMBEDDING_FILES = ("train_features.npy", "val_features.npy", "val_labels.npy", "val_correct.npy",
                   "val_confidence.npy")
METRICS = ("auroc_err", "ap_err", "auroc_lowconf", "ap_lowconf")


@dataclass(frozen=True)
class Contrast:
    """One comparison: bigger grid minus smaller grid on one endpoint metric."""

    big: int
    small: int
    metric: str


@dataclass(frozen=True)
class Reproduction:
    """Where the original run's maps and results are, for the reproduction checks (MNIST only)."""

    maps_dir: Path
    maps_dir_config: str          # as written in the config, for the records
    model_pattern: str
    results: Path
    saved_maps_tolerance: float
    retrain_auroc_tolerance: float

    def map_path(self, grid: int, nb: int, epochs: int, b: int) -> Path:
        return self.maps_dir / self.model_pattern.format(grid=grid, nb=nb, epochs=epochs, b=b)


@dataclass(frozen=True)
class GridSettings:
    """Everything the stages need."""

    config_path: Path
    run_id: str
    output_dir: Path
    protocol_path: Path
    input_configs: dict[str, Path]
    embeddings_dir: Path
    num_classes: int
    grids: tuple[int, ...]
    candidate_fractions: tuple[float, ...]
    epochs: int
    tune_seed: int
    bootstrap_maps: int
    resample_seed_base: int
    reps: int
    split_seed_base: int
    margin: float
    low_confidence_quantile: float
    primary: Contrast
    secondary: tuple[Contrast, ...]
    exploratory: tuple[Contrast, ...]
    expected: dict[str, int]
    reproduction: Reproduction | None = None

    def candidates(self, grid: int) -> tuple[int, ...]:
        """Stage 1 neighbourhoods: the fixed fractions of the side length, rounded."""
        return tuple(int(round(f * grid)) for f in self.candidate_fractions)

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def records_dir(self) -> Path:
        return self.output_dir / "records"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def inputs_path(self) -> Path:
        return self.output_dir / "metadata" / "inputs.json"

    @property
    def choice_path(self) -> Path:
        return self.output_dir / "stage1_choice.json"

    @property
    def results_path(self) -> Path:
        return self.output_dir / "results.json"

    @property
    def figure_path(self) -> Path:
        return self.output_dir / "pdf" / "grid_comparison_results.pdf"


def tune_id(grid: int, nb: int, epochs: int, seed: int) -> str:
    return f"tune_{grid}x{grid}_nb{nb}_ep{epochs}_s{seed}"


def boot_id(grid: int, nb: int, epochs: int, b: int) -> str:
    return f"boot_{grid}x{grid}_nb{nb}_ep{epochs}_b{b}"


def _contrast(raw: dict[str, Any]) -> Contrast:
    metric = str(raw["metric"])
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {METRICS}.")
    return Contrast(int(raw["big"]), int(raw["small"]), metric)


def load_grid_settings(config_path: str | Path | None, repo_root: Path) -> GridSettings:
    """Read the config and check the design."""
    cfg_path = resolve_path(repo_root, config_path or DEFAULT_CONFIG)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    inputs = {k: resolve_path(repo_root, v) for k, v in raw["inputs"].items()}
    som_raw = yaml.safe_load(inputs["som_config"].read_text(encoding="utf-8"))
    cnn = load_config(str(inputs["cnn_config"]))
    design, endpoint, comp = raw["design"], raw["endpoint"], raw["comparisons"]

    grids = tuple(int(g) for g in design["grids"])
    contrasts = [_contrast(comp["primary"]), *map(_contrast, comp["secondary"]), *map(_contrast, comp["exploratory"])]
    for c in contrasts:
        if c.big not in grids or c.small not in grids or c.big == c.small:
            raise ValueError(f"Contrast {c} must compare two different grids of {grids}.")
    name, revision = str(raw["run"]["name"]), str(raw["run"].get("revision") or "")
    run_id = check_run_id(
        f"{name}_g{'_'.join(map(str, grids))}_b{int(design['bootstrap_maps'])}_r{int(endpoint['reps'])}"
        + (f"_{revision}" if revision else ""))
    rep = raw.get("reproduction")
    return GridSettings(
        config_path=cfg_path,
        run_id=run_id,
        output_dir=resolve_path(repo_root, raw["paths"]["output_root"]) / run_id,
        protocol_path=resolve_path(repo_root, raw["protocol"]["document"]),
        input_configs=inputs,
        embeddings_dir=resolve_path(repo_root, som_raw["paths"]["embeddings_dir"]),
        num_classes=int(cnn.dataset.num_classes),
        grids=grids,
        candidate_fractions=tuple(float(f) for f in design["candidate_fractions"]),
        epochs=int(design["epochs"]),
        tune_seed=int(design["tune_seed"]),
        bootstrap_maps=int(design["bootstrap_maps"]),
        resample_seed_base=int(design["resample_seed_base"]),
        reps=int(endpoint["reps"]),
        split_seed_base=int(endpoint["split_seed_base"]),
        margin=float(endpoint["margin"]),
        low_confidence_quantile=float(endpoint["low_confidence_quantile"]),
        primary=contrasts[0],
        secondary=tuple(contrasts[1:1 + len(comp["secondary"])]),
        exploratory=tuple(contrasts[1 + len(comp["secondary"]):]),
        expected={k: int(v) for k, v in raw["expected"].items()},
        reproduction=None if not rep else Reproduction(
            maps_dir=resolve_path(repo_root, rep["maps_dir"]),
            maps_dir_config=str(rep["maps_dir"]),
            model_pattern=str(rep["model_pattern"]),
            results=resolve_path(repo_root, rep["results"]),
            saved_maps_tolerance=float(rep["saved_maps_tolerance"]),
            retrain_auroc_tolerance=float(rep["retrain_auroc_tolerance"]),
        ),
    )
