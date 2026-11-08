"""Settings of the SOM re-analysis: read from som_reanalysis.yaml, retraining.yaml and som.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.settings import (
    resolve_path,
)
from src.v1_mnist.component.utils.retraining_settings import (
    ExperimentSettings,
    load_experiment_settings,
)

DEFAULT_CONFIG = "src/v1_mnist/component/configs/som_reanalysis.yaml"


@dataclass(frozen=True)
class SomRecipe:
    """The Phase 2 SOM training recipe."""

    grid_height: int
    grid_width: int
    init_neighborhood: int
    epochs: int
    steps: int
    seed: int
    history_every: int
    hotspot_fraction: float
    min_hotspots: int


@dataclass(frozen=True)
class ReanalysisSettings:
    """Everything the three stages need."""

    config_path: Path
    run_id: str
    output_dir: Path
    protocol_path: Path
    retraining: ExperimentSettings
    recipe: SomRecipe
    purity_cutoff: float
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    reference: str
    layer: str
    batch_size: int
    device: str
    min_agreement: float
    adjacency: float
    pairs: tuple[tuple[str, str], ...]
    confidence: float
    tolerance_pct: float
    example_seed: int

    @property
    def cells(self) -> tuple[str, ...]:
        """Cell names 'arm_seedK' in a fixed order."""
        return tuple(cell_name(a, s) for a in self.arms for s in self.seeds)

    def folder(self, kind: str) -> Path:
        return self.output_dir / kind

    @property
    def inputs_path(self) -> Path:
        return self.output_dir / "metadata" / "inputs.json"


def cell_name(arm: str, seed: int) -> str:
    return f"{arm}_seed{seed}"


def embeddings_folder(settings: ReanalysisSettings, cell: str) -> Path:
    """Where the embeddings of one cell are stored."""
    return settings.folder("embeddings") / cell


def load_reanalysis_settings(config_path: str | Path | None, repo_root: Path) -> ReanalysisSettings:
    """Read the configs and check the cell list against the retraining run."""
    cfg_path = resolve_path(repo_root, config_path or DEFAULT_CONFIG)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    retraining = load_experiment_settings(
        raw["inputs"]["retraining_section"], raw["inputs"]["retraining_config"], repo_root
    )
    som_raw = yaml.safe_load(retraining.analysis.config_path.read_text(encoding="utf-8"))
    retr_raw = yaml.safe_load(retraining.config_path.read_text(encoding="utf-8"))

    arms = tuple(str(a) for a in raw["cells"]["arms"])
    seeds = tuple(int(s) for s in raw["cells"]["seeds"])
    known_arms = {a.name for a in retraining.arms}
    missing = [a for a in arms if a not in known_arms]
    if missing:
        raise ValueError(f"Arms not in the {retraining.section} run: {missing}")
    missing_seeds = [s for s in seeds if s not in retraining.seeds]
    if missing_seeds:
        raise ValueError(f"Seeds not in the {retraining.section} run: {missing_seeds}")
    pairs = tuple((str(a), str(b)) for a, b in raw["comparison"]["pairs"])
    for first, second in pairs:
        if first not in arms or second not in arms:
            raise ValueError(f"Pair ({first}, {second}) uses an arm that is not in cells.arms.")

    revision = raw["run"].get("revision") or ""
    run_id = f"{raw['run']['name']}_{retraining.analysis.selected_model}" + (f"_{revision}" if revision else "")
    som, ana = som_raw["som"], som_raw["analysis"]
    return ReanalysisSettings(
        config_path=cfg_path,
        run_id=run_id,
        output_dir=resolve_path(repo_root, raw["paths"]["output_root"]) / run_id,
        protocol_path=resolve_path(repo_root, raw["protocol"]["document"]),
        retraining=retraining,
        recipe=SomRecipe(
            grid_height=int(som["grid_height"]),
            grid_width=int(som["grid_width"]),
            init_neighborhood=int(som["init_neighborhood"]),
            epochs=int(som["epochs"]),
            steps=int(som["steps"]),
            seed=int(som["random_seed"]),
            history_every=int(som.get("qe_history_every", 1)),
            hotspot_fraction=float(ana["hotspot_fraction"]),
            min_hotspots=int(ana["min_hotspots"]),
        ),
        purity_cutoff=float(retr_raw["pools"]["purity_cutoff"]),
        arms=arms,
        seeds=seeds,
        reference=str(raw["cells"]["reference"]),
        layer=str(raw["extraction"]["layer"]),
        batch_size=int(raw["extraction"]["batch_size"]),
        device=str(raw["extraction"]["device"]),
        min_agreement=float(raw["extraction"]["min_prediction_agreement"]),
        adjacency=float(raw["measures"]["adjacency_distance"]),
        pairs=pairs,
        confidence=float(raw["comparison"]["confidence"]),
        tolerance_pct=float(raw["comparison"]["organisation_tolerance_pct"]),
        example_seed=int(raw["figures"]["example_seed"]),
    )
