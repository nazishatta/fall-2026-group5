"""Settings of the held-out-class novelty step: held_out_class_novelty.yaml plus the configs it names.

Protocol: docs/p3_novelty_detection/held_out_class_novelty/HELD_OUT_CLASS_NOVELTY_PROTOCOL.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.settings import resolve_path
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.image_datasets import DatasetSpec, dataset_spec
from src.v1_mnist.component.utils.retraining_settings import TrainSettings

DEFAULT_CONFIG = "src/v1_mnist/component/configs/held_out_class_novelty.yaml"


@dataclass(frozen=True)
class SomRecipe:
    """The Phase 2 SOM recipe (som.yaml)."""

    grid_height: int
    grid_width: int
    init_neighborhood: int
    epochs: int
    steps: int
    seed: int


@dataclass(frozen=True)
class HeldOutSettings:
    """Everything the two stages need."""

    config_path: Path
    run_id: str
    output_dir: Path
    protocol_path: Path
    cnn_config_path: Path
    som_config_path: Path
    data_root: Path
    embeddings_dir: Path
    num_classes: int
    unknown_digits: tuple[tuple[int, ...], ...]
    train: TrainSettings
    training_seed: int
    device: str
    layer: str
    batch_size: int
    som: SomRecipe
    energy_temperature: float
    threshold_percentile: float
    knn_neighbors: int
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence: float
    auroc_margin: float
    null_iterations: int
    null_seed: int
    coverage_share: float
    dataset: DatasetSpec = DatasetSpec()   # dataset and network input size (Phase 1 config)

    @property
    def split_ids(self) -> tuple[int, ...]:
        """Splits are numbered from 1, in config order."""
        return tuple(range(1, len(self.unknown_digits) + 1))

    def split_dir(self, split: int) -> Path:
        return self.output_dir / "splits" / f"split{split}"

    def metrics_path(self, split: int) -> Path:
        return self.split_dir(split) / "metrics.json"

    @property
    def inputs_path(self) -> Path:
        return self.output_dir / "metadata" / "inputs.json"


def draw_splits(seed: int, num_classes: int, unknown: int, count: int) -> tuple[tuple[int, ...], ...]:
    """The protocol's rule: draw ``unknown`` classes without replacement, sorted, until ``count`` distinct sets."""
    rng = np.random.default_rng(seed)
    out: list[tuple[int, ...]] = []
    while len(out) < count:
        draw = tuple(sorted(int(c) for c in rng.choice(num_classes, unknown, replace=False)))
        if draw not in out:
            out.append(draw)
    return tuple(out)


def run_id_for(name: str, som: SomRecipe, revision: str) -> str:
    base = (f"{name}_lenet5_som{som.grid_height}x{som.grid_width}_nb{som.init_neighborhood}"
            f"_ep{som.epochs}_s{som.seed}")
    return base + (f"_{revision}" if revision else "")


def load_held_out_settings(config_path: str | Path | None, repo_root: Path) -> HeldOutSettings:
    """Read the configs and check the splits against the protocol's drawing rule."""
    cfg_path = resolve_path(repo_root, config_path or DEFAULT_CONFIG)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cnn_path = resolve_path(repo_root, raw["inputs"]["cnn_config"])
    som_path = resolve_path(repo_root, raw["inputs"]["som_config"])
    cnn = load_config(str(cnn_path))
    som_raw = yaml.safe_load(som_path.read_text(encoding="utf-8"))
    num_classes = int(cnn.dataset.num_classes)

    sp = raw["splits"]
    unknown = tuple(tuple(int(d) for d in s) for s in sp["unknown_digits"])
    redrawn = draw_splits(int(sp["generation_seed"]), num_classes, int(sp["unknown_per_split"]), int(sp["count"]))
    if unknown != redrawn:
        raise ValueError(f"splits.unknown_digits differ from the protocol's draw: {redrawn}")

    s = som_raw["som"]
    som = SomRecipe(grid_height=int(s["grid_height"]), grid_width=int(s["grid_width"]),
                    init_neighborhood=int(s["init_neighborhood"]), epochs=int(s["epochs"]),
                    steps=int(s["steps"]), seed=int(s["random_seed"]))
    run_id = run_id_for(str(raw["run"]["name"]), som, str(raw["run"].get("revision") or ""))
    stats = raw["statistics"]
    return HeldOutSettings(
        config_path=cfg_path,
        run_id=run_id,
        output_dir=resolve_path(repo_root, raw["paths"]["output_root"]) / run_id,
        protocol_path=resolve_path(repo_root, raw["protocol"]["document"]),
        cnn_config_path=cnn_path,
        som_config_path=som_path,
        data_root=resolve_path(repo_root, cnn.paths.data_root),
        dataset=dataset_spec(cnn),
        # Checked when the run starts (provenance hashes the split files), not here.
        embeddings_dir=resolve_path(repo_root, som_raw["paths"]["embeddings_dir"]),
        num_classes=num_classes,
        unknown_digits=unknown,
        train=TrainSettings(
            epochs=int(cnn.training.epochs), lr=float(cnn.training.lr),
            weight_decay=float(cnn.training.weight_decay), step_size=int(cnn.training.step_size),
            gamma=float(cnn.training.gamma), batch_size=int(cnn.dataloader.batch_size),
            num_classes=num_classes - int(sp["unknown_per_split"]),
        ),
        training_seed=int(raw["training"]["seed"]),
        device=str(raw["training"]["device"]),
        layer=str(raw["extraction"]["layer"]),
        batch_size=int(raw["extraction"]["batch_size"]),
        som=som,
        energy_temperature=float(raw["scores"]["energy_temperature"]),
        threshold_percentile=float(raw["scores"]["threshold_percentile"]),
        knn_neighbors=int(raw["scores"]["knn_neighbors"]),
        bootstrap_resamples=int(stats["bootstrap_resamples"]),
        bootstrap_seed=int(stats["bootstrap_seed"]),
        confidence=float(stats["confidence"]),
        auroc_margin=float(stats["auroc_margin"]),
        null_iterations=int(stats["null_iterations"]),
        null_seed=int(stats["null_seed"]),
        coverage_share=float(raw["map"]["coverage_share"]),
    )
