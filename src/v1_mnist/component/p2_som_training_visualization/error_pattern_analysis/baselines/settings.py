"""Settings of the error pattern baseline check.

Read from the check's own config and from the headline SOM config (grid size, selected SOM,
embeddings folder, hotspot rule, split and shuffle settings), so nothing is hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.error_pattern_robustness import (
    RobustnessSettings,
    robustness_settings,
)
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.run_settings import (
    AnalysisSettings,
    analysis_config,
    load_analysis_settings,
)

DEFAULT_CONFIG = "src/v1_mnist/component/configs/error_pattern_baselines.yaml"


@dataclass(frozen=True)
class BaselineSettings:
    """Everything the baseline check needs besides the headline SOM's own settings."""

    config_path: Path
    som_config: str
    run_id: str
    output_dir: Path
    work_dir: Path
    num_cells: int
    grid: tuple[int, int]
    feature_range: tuple[float, float]
    kmeans_seeds: tuple[int, ...]
    kmeans_n_init: int
    knn_neighbors: int
    detection_repeats: int
    bootstrap_resamples: int
    bootstrap_seed: int
    robustness: RobustnessSettings

    @property
    def results_path(self) -> Path:
        return self.output_dir / "results.json"


def resolve_path(repo_root: Path, path: str | Path) -> Path:
    """Absolute path; relative paths start at the code folder (the one holding src/)."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (repo_root / candidate).resolve()


def load_baseline_settings(
    config_path: str | Path | None, repo_root: Path
) -> tuple[BaselineSettings, AnalysisSettings]:
    """Read the check's config and the headline SOM's settings it points to."""
    cfg_path = resolve_path(repo_root, config_path or DEFAULT_CONFIG)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    som_config = str(raw["inputs"]["som_config"])
    analysis = load_analysis_settings(som_config, repo_root)
    som = load_config(str(resolve_path(repo_root, som_config)))

    # Short run ID: the grid is in the name (the exact SOM model and its SHA-256 are recorded in results.json).
    # The first run has no suffix; a new or re-tuned run sets run.revision (r2, r3, ...) and gets its own folder.
    revision = raw["run"].get("revision") or ""
    run_id = f"{raw['run']['name']}_som_{analysis.grid_height}x{analysis.grid_width}" + (f"_{revision}" if revision else "")
    output_dir = resolve_path(repo_root, raw["paths"]["output_root"]) / run_id
    settings = BaselineSettings(
        config_path=cfg_path,
        som_config=som_config,
        run_id=run_id,
        output_dir=output_dir,
        work_dir=output_dir / "work",
        num_cells=analysis.num_neurons,
        grid=(analysis.grid_height, analysis.grid_width),
        feature_range=(float(som.preprocessing.feature_range[0]), float(som.preprocessing.feature_range[1])),
        kmeans_seeds=tuple(int(s) for s in raw["baselines"]["kmeans"]["seeds"]),
        kmeans_n_init=int(raw["baselines"]["kmeans"]["n_init"]),
        knn_neighbors=int(raw["baselines"]["knn"]["neighbors"]),
        detection_repeats=int(raw["error_detection"]["repeats"]),
        bootstrap_resamples=int(raw["error_detection"]["bootstrap"]["resamples"]),
        bootstrap_seed=int(raw["error_detection"]["bootstrap"]["seed"]),
        robustness=robustness_settings(analysis_config(som_config, repo_root)),
    )
    return settings, analysis
