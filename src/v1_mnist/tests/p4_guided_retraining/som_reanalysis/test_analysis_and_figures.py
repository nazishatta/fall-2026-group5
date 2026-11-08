"""The analysis stage and the figures on small synthetic cells (no PyTorch, no real run)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.analysis import (
    run_analysis,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.reanalysis_figures import (
    make_figures,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.settings import (
    cell_name,
)

ARMS = ("standard", "augmented_som", "augmented_random")
SEEDS = (1, 2, 3, 4, 5)
N, SIDE, CLASSES = 300, 4, 3


def build(tmp_path: Path) -> Any:
    rng = np.random.default_rng(0)
    emb = tmp_path / "phase1"
    emb.mkdir()
    labels = rng.integers(0, CLASSES, N)
    np.save(emb / "val_labels.npy", labels)
    np.save(emb / "val_preds.npy", labels)
    pos = np.array([[i % SIDE for i in range(SIDE * SIDE)], [i // SIDE for i in range(SIDE * SIDE)]], dtype=float)
    dist = np.sqrt(((pos[:, :, None] - pos[:, None, :]) ** 2).sum(axis=0))

    def settings_folder(kind: str) -> Path:
        return tmp_path / "run" / kind

    cells = ["baseline", *[cell_name(a, s) for a in ARMS for s in SEEDS]]
    for cell in cells:
        (settings_folder("assignments")).mkdir(parents=True, exist_ok=True)
        (settings_folder("metrics")).mkdir(parents=True, exist_ok=True)
        bmu = (labels * 5 + rng.integers(0, 3, N)) % (SIDE * SIDE)
        np.savez(settings_folder("assignments") / f"{cell}.npz", val_bmu=bmu, neuron_dist=dist, pos=pos)
        quality = {"topological_error_1st_order_pct": 1.0 + rng.random(), "quantization_error": 1.0}
        (settings_folder("metrics") / f"{cell}.json").write_text(json.dumps({"validation_quality": quality}))
        if cell != "baseline":
            folder = settings_folder("embeddings") / cell
            folder.mkdir(parents=True, exist_ok=True)
            preds = np.where(rng.random(N) < 0.05, (labels + 1) % CLASSES, labels)
            np.save(folder / "val_preds.npy", preds)
            np.save(folder / "val_features.npy", rng.normal(size=(N, 8)))
    return SimpleNamespace(
        folder=settings_folder, reference="baseline", arms=ARMS, seeds=SEEDS,
        cells=tuple(c for c in cells if c != "baseline"),
        recipe=SimpleNamespace(hotspot_fraction=0.05, min_hotspots=2),
        retraining=SimpleNamespace(analysis=SimpleNamespace(embeddings_dir=emb),
                                   train=SimpleNamespace(num_classes=CLASSES)),
        purity_cutoff=0.95, adjacency=1.1, confidence=0.95, tolerance_pct=1.0, example_seed=1,
        pairs=(("augmented_som", "standard"), ("augmented_som", "augmented_random")),
    )


def test_analysis_writes_tables_and_figures_are_pdf(tmp_path: Path) -> None:
    settings = build(tmp_path)
    results = run_analysis(settings)
    tables = settings.folder("tables")
    for name in ("cell_measures.csv", "paired_comparisons.csv", "statements.json", "cka.json"):
        assert (tables / name).is_file()
    assert set(results["statements"]) == {
        "R1_boundaries_sharpen_vs_standard", "R2_hotspots_shrink_vs_standard",
        "R3_guidance_sharpens_vs_random", "R4_guidance_shrinks_hotspots_vs_random",
        "R5_map_not_less_organised_vs_standard",
    }
    assert len(results["table"]) == 16
    paths = make_figures(settings, results)
    assert all(p.suffix == ".pdf" and p.is_file() for p in paths)
    assert not list(settings.folder("figures").glob("*.png"))
