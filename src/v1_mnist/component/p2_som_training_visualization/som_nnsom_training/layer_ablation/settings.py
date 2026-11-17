"""Settings of the feature-layer ablation: feature_layer_ablation.yaml plus the configs it names.

Protocol: docs/p2_som_training_visualization/som_nnsom_training/FEATURE_LAYER_ABLATION_PROTOCOL.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.settings import resolve_path
from src.v1_mnist.component.utils.config import load_config

DEFAULT_CONFIG = "src/v1_mnist/component/configs/feature_layer_ablation.yaml"
LAYER_DIMS = {"fc1": 120, "fc2": 84}   # LeNet-5 (lenet5.py)
REFERENCE_LAYER = "fc2"                # the Phase 1 layer, checked against the frozen embeddings


@dataclass(frozen=True)
class SomRecipe:
    """The Phase 2 SOM recipe (som.yaml), used for every layer."""

    grid_height: int
    grid_width: int
    init_neighborhood: int
    epochs: int
    steps: int

    @property
    def num_neurons(self) -> int:
        return self.grid_height * self.grid_width


@dataclass(frozen=True)
class AblationSettings:
    """Everything the three stages need."""

    config_path: Path
    run_id: str
    output_dir: Path
    protocol_path: Path
    input_configs: dict[str, Path]
    data_root: Path
    embeddings_dir: Path
    checkpoint: Path
    checkpoint_sha256: str
    svhn_file: Path
    svhn_sha256: str
    layers: tuple[str, ...]
    som_seeds: tuple[int, ...]
    recipe: SomRecipe
    num_classes: int
    hotspot_fraction: float
    min_hotspots: int
    detection_repeats: int
    split_seed_base: int
    batch_size: int
    device: str
    image_size: tuple[int, int]
    fc2_tolerance: float
    confidence: float

    def folder(self, kind: str) -> Path:
        return self.output_dir / kind

    def embedding_path(self, layer: str, split: str) -> Path:
        return self.folder("embeddings") / layer / f"{split}_features.npy"

    def metrics_path(self, layer: str, seed: int) -> Path:
        return self.folder("metrics") / f"{cell_name(layer, seed)}.json"

    @property
    def inputs_path(self) -> Path:
        return self.output_dir / "metadata" / "inputs.json"

    @property
    def extraction_path(self) -> Path:
        return self.folder("embeddings") / "extraction.json"


def cell_name(layer: str, seed: int) -> str:
    return f"{layer}_seed{seed}"


def run_id_for(name: str, recipe: SomRecipe, seeds: int, revision: str) -> str:
    base = (f"{name}_som{recipe.grid_height}x{recipe.grid_width}_nb{recipe.init_neighborhood}"
            f"_ep{recipe.epochs}_seeds{seeds}")
    return base + (f"_{revision}" if revision else "")


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_ablation_settings(config_path: str | Path | None, repo_root: Path) -> AblationSettings:
    """Read the configs and check the layers and seeds."""
    cfg_path = resolve_path(repo_root, config_path or DEFAULT_CONFIG)
    raw = _yaml(cfg_path)
    paths = {k: resolve_path(repo_root, v) for k, v in raw["inputs"].items()}
    cnn = load_config(str(paths["cnn_config"]))
    som_raw, base_raw, prov = _yaml(paths["som_config"]), _yaml(paths["baselines_config"]), _yaml(
        paths["phase3_provenance_config"])

    layers = tuple(str(x) for x in raw["layers"])
    if set(layers) != set(LAYER_DIMS) or len(layers) != 2 or layers[1] != REFERENCE_LAYER:
        raise ValueError(f"layers must be [fc1, fc2] (compared as fc1 - fc2), got {list(layers)}.")
    seeds = tuple(int(s) for s in raw["som_seeds"])
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("som_seeds needs at least 2 distinct seeds.")
    if int(som_raw["som"]["random_seed"]) in seeds:
        raise ValueError("The headline SOM seed must not be reused (protocol section 2).")

    s = som_raw["som"]
    recipe = SomRecipe(int(s["grid_height"]), int(s["grid_width"]), int(s["init_neighborhood"]),
                       int(s["epochs"]), int(s["steps"]))
    run_id = run_id_for(str(raw["run"]["name"]), recipe, len(seeds), str(raw["run"].get("revision") or ""))
    ana, ext = som_raw["analysis"], raw["extraction"]
    return AblationSettings(
        config_path=cfg_path,
        run_id=run_id,
        output_dir=resolve_path(repo_root, raw["paths"]["output_root"]) / run_id,
        protocol_path=resolve_path(repo_root, raw["protocol"]["document"]),
        input_configs=paths,
        data_root=resolve_path(repo_root, cnn.paths.data_root),
        embeddings_dir=resolve_path(repo_root, som_raw["paths"]["embeddings_dir"]),
        checkpoint=resolve_path(repo_root, prov["inputs"]["checkpoint"]),
        checkpoint_sha256=str(prov["expected_sha256"]["checkpoint"]),
        svhn_file=resolve_path(repo_root, prov["svhn_file"]),
        svhn_sha256=str(prov["expected_sha256"]["svhn_file"]),
        layers=layers,
        som_seeds=seeds,
        recipe=recipe,
        num_classes=int(cnn.dataset.num_classes),
        hotspot_fraction=float(ana["hotspot_fraction"]),
        min_hotspots=int(ana["min_hotspots"]),
        detection_repeats=int(base_raw["error_detection"]["repeats"]),
        split_seed_base=int(ana["robustness"]["split_seed_base"]),
        batch_size=int(ext["batch_size"]),
        device=str(ext["device"]),
        image_size=(int(ext["image_size"][0]), int(ext["image_size"][1])),
        fc2_tolerance=float(ext["fc2_tolerance"]),
        confidence=float(raw["statistics"]["confidence"]),
    )
