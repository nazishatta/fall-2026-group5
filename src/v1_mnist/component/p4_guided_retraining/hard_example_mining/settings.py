"""Settings of the hard example mining step: which training images count as hard.

Read from retraining.yaml and from the headline SOM config (selected SOM, embeddings folder,
hotspot rule), so nothing is hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.v1_mnist.component.utils.hotspots import n_hotspots
from src.v1_mnist.component.utils.run_settings import (
    AnalysisSettings,
    analysis_config,
    load_analysis_settings,
)

DEFAULT_CONFIG = "src/v1_mnist/component/configs/retraining.yaml"


@dataclass(frozen=True)
class PoolSettings:
    """Everything the pool step needs besides the headline SOM's own settings."""

    config_path: Path
    som_config: str
    run_id: str
    output_dir: Path
    purity_cutoff: float
    qe_percentile: float
    num_hotspots: int

    @property
    def results_path(self) -> Path:
        return self.output_dir / "results.json"


def resolve_path(repo_root: Path, path: str | Path) -> Path:
    """Absolute path; relative paths start at the code folder (the one holding src/)."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (repo_root / candidate).resolve()


def load_pool_settings(
    config_path: str | Path | None, repo_root: Path
) -> tuple[PoolSettings, AnalysisSettings]:
    """Read retraining.yaml and the headline SOM's settings it points to."""
    cfg_path = resolve_path(repo_root, config_path or DEFAULT_CONFIG)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    som_config = str(raw["inputs"]["som_config"])
    analysis = load_analysis_settings(som_config, repo_root)
    rule = analysis_config(som_config, repo_root)

    revision = raw["run"].get("revision") or ""
    run_id = f"{raw['run']['name']}_{analysis.selected_model}" + (f"_{revision}" if revision else "")
    settings = PoolSettings(
        config_path=cfg_path,
        som_config=som_config,
        run_id=run_id,
        output_dir=resolve_path(repo_root, raw["paths"]["output_root"]) / run_id,
        purity_cutoff=float(raw["pools"]["purity_cutoff"]),
        qe_percentile=float(raw["pools"]["qe_percentile"]),
        num_hotspots=n_hotspots(
            analysis.num_neurons, float(rule.hotspot_fraction), int(rule.min_hotspots)
        ),
    )
    return settings, analysis
