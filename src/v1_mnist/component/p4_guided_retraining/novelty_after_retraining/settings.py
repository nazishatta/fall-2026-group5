"""Settings of the novelty-after-retraining step: novelty_after_retraining.yaml plus the configs it names.

Protocol: docs/p4_guided_retraining/novelty_after_retraining/NOVELTY_AFTER_RETRAINING_PROTOCOL.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.settings import resolve_path
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.settings import (
    ReanalysisSettings,
    cell_name,
    load_reanalysis_settings,
)

DEFAULT_CONFIG = "src/v1_mnist/component/configs/novelty_after_retraining.yaml"


@dataclass(frozen=True)
class NoveltySettings:
    """Everything the two stages need."""

    config_path: Path
    run_id: str
    output_dir: Path
    protocol_path: Path
    reanalysis: ReanalysisSettings
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    reference: str
    svhn_file: Path
    svhn_sha256: str
    image_size: tuple[int, int]
    batch_size: int
    device: str
    energy_temperature: float
    threshold_percentile: float
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence: float
    auroc_margin: float
    pairs: tuple[tuple[str, str], ...]

    @property
    def cells(self) -> tuple[str, ...]:
        """Retrained cells 'arm_seedK' in a fixed order (the reference is separate)."""
        return tuple(cell_name(a, s) for a in self.arms for s in self.seeds)

    def folder(self, kind: str) -> Path:
        return self.output_dir / kind

    @property
    def inputs_path(self) -> Path:
        return self.output_dir / "metadata" / "inputs.json"


def load_novelty_settings(config_path: str | Path | None, repo_root: Path) -> NoveltySettings:
    """Read the configs and check the cells against the SOM re-analysis run."""
    cfg_path = resolve_path(repo_root, config_path or DEFAULT_CONFIG)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    reanalysis = load_reanalysis_settings(raw["inputs"]["som_reanalysis_config"], repo_root)
    provenance = yaml.safe_load(
        resolve_path(repo_root, raw["inputs"]["phase3_provenance_config"]).read_text(encoding="utf-8")
    )
    arms = tuple(str(a) for a in raw["cells"]["arms"])
    seeds = tuple(int(s) for s in raw["cells"]["seeds"])
    if not set(arms) <= set(reanalysis.arms) or not set(seeds) <= set(reanalysis.seeds):
        raise ValueError("Every arm and seed must be a cell of the SOM re-analysis run.")
    if str(raw["cells"]["reference"]) != reanalysis.reference:
        raise ValueError("The reference cell must be the SOM re-analysis reference.")
    pairs = tuple((str(a), str(b)) for a, b in raw["comparison"]["pairs"])
    if any(a not in arms or b not in arms for a, b in pairs):
        raise ValueError("Every comparison pair must use arms from cells.arms.")
    if raw["ood"]["dataset"] != "svhn":
        raise ValueError("Only SVHN is defined as the novel set in the protocol.")

    revision = raw["run"].get("revision") or ""
    selected = reanalysis.retraining.analysis.selected_model
    run_id = f"{raw['run']['name']}_{selected}" + (f"_{revision}" if revision else "")
    stats = raw["statistics"]
    return NoveltySettings(
        config_path=cfg_path,
        run_id=run_id,
        output_dir=resolve_path(repo_root, raw["paths"]["output_root"]) / run_id,
        protocol_path=resolve_path(repo_root, raw["protocol"]["document"]),
        reanalysis=reanalysis,
        arms=arms,
        seeds=seeds,
        reference=reanalysis.reference,
        svhn_file=resolve_path(repo_root, provenance["svhn_file"]),
        svhn_sha256=str(provenance["expected_sha256"]["svhn_file"]),
        image_size=(int(raw["ood"]["image_size"][0]), int(raw["ood"]["image_size"][1])),
        batch_size=int(raw["ood"]["batch_size"]),
        device=str(raw["ood"]["device"]),
        energy_temperature=float(raw["scores"]["energy_temperature"]),
        threshold_percentile=float(raw["scores"]["qe_threshold_percentile"]),
        bootstrap_resamples=int(stats["bootstrap_resamples"]),
        bootstrap_seed=int(stats["bootstrap_seed"]),
        confidence=float(stats["confidence"]),
        auroc_margin=float(stats["auroc_margin"]),
        pairs=pairs,
    )
