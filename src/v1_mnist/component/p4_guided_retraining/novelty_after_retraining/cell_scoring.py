"""Stage 1: score one cell (protocol sections 2 to 4). Needs PyTorch.

For each cell: load the network and the cell's SOM (both SHA-256 checked), reuse the saved training and
validation embeddings, embed the SVHN test images with the same network, and compute S1 to S4 and D1.
Every reused input is checked: the reconstructed predictions must equal the saved ones, and the SOM must
reproduce the re-analysis distances exactly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import torch

from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (
    FeatureExtractor,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
    svhn_test_loader,
)
from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.provenance import (
    arm_and_seed,
    start_run,
    verify_run,
)
from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.scoring import (
    cell_metrics,
    relu_logits,
    score_split,
)
from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.settings import (
    NoveltySettings,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.settings import (
    embeddings_folder,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.som_cell import (
    assign,
    model_path,
)
from src.v1_mnist.component.utils.cnn_retraining import load_model, resolve_device
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.run_settings import verify_selected_model
from src.v1_mnist.component.utils.som_io import (
    load_selected_som,
    load_split,
    sha256_file,
)

log = logging.getLogger("v1_mnist.novelty_after_retraining")


def _inputs_for(settings: NoveltySettings, cell: str) -> dict[str, Any]:
    """Network checkpoint, SOM, embeddings and the re-analysis distances of one cell."""
    re, retr = settings.reanalysis, settings.reanalysis.retraining
    ana = retr.analysis
    if cell == settings.reference:
        verify_selected_model(ana)
        return {"checkpoint": retr.cnn_checkpoint, "som_path": ana.model_path,
                "train_x": load_split("train", ana.embeddings_dir, ana.feature_dim)["features"],
                "val_x": load_split("val", ana.embeddings_dir, ana.feature_dim)["features"],
                "val_pred": np.load(ana.embeddings_dir / "val_preds.npy"),
                "val_dist": np.load(re.folder("assignments") / f"{cell}.npz")["val_dist"]}
    record = json.loads((re.folder("metrics") / f"{cell}.json").read_text(encoding="utf-8"))
    emb = embeddings_folder(re, cell)
    extraction = json.loads((emb / "extraction.json").read_text(encoding="utf-8"))
    checks = {
        "SOM model": (sha256_file(model_path(re, cell)), record["model_sha256"]),
        "training embeddings": (sha256_file(emb / "train_features.npy"), extraction["train_features_sha256"]),
        "validation embeddings": (sha256_file(emb / "val_features.npy"), extraction["val_features_sha256"]),
        "checkpoint": (sha256_file(retr.checkpoint_path(*arm_and_seed(cell))), extraction["checkpoint_sha256"]),
    }
    bad = [name for name, (seen, recorded) in checks.items() if seen != recorded]
    if bad:
        raise RuntimeError(f"{cell}: differs from the SOM re-analysis record: {bad}")
    return {"checkpoint": retr.checkpoint_path(*arm_and_seed(cell)), "som_path": model_path(re, cell),
            "train_x": np.load(emb / "train_features.npy"), "val_x": np.load(emb / "val_features.npy"),
            "val_pred": np.load(emb / "val_preds.npy"),
            "val_dist": np.load(re.folder("assignments") / f"{cell}.npz")["val_dist"]}


def _svhn_features(model: torch.nn.Module, settings: NoveltySettings, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = svhn_test_loader(settings.svhn_file, settings.image_size, settings.batch_size)
    features, _labels, predictions = FeatureExtractor(model, settings.reanalysis.layer).extract(loader, device)
    return features.astype(np.float32), predictions


def score_cell(settings: NoveltySettings, cell: str, device: torch.device) -> None:
    """Write scores (NPZ) and metrics (JSON) of one cell, or skip it if its metrics exist."""
    out = settings.folder("metrics") / f"{cell}.json"
    if out.is_file():
        log.info("%s: already scored, skipped.", cell)
        return
    retr = settings.reanalysis.retraining
    inputs = _inputs_for(settings, cell)
    model = load_model(torch.load(inputs["checkpoint"], map_location="cpu"), retr.train.num_classes, device,
                       retr.train.in_channels, retr.train.image_size)
    w = model.fc3.weight.detach().cpu().numpy()
    b = model.fc3.bias.detach().cpu().numpy()
    svhn_x, svhn_pred = _svhn_features(model, settings, device)
    logits = {k: relu_logits(x, w, b) for k, x in
              (("train", inputs["train_x"]), ("normal", inputs["val_x"]), ("novel", svhn_x))}
    if not np.array_equal(logits["normal"].argmax(1), inputs["val_pred"]):
        raise RuntimeError(f"{cell}: reconstructed validation predictions differ from the saved ones.")
    if not np.array_equal(logits["novel"].argmax(1), svhn_pred):
        raise RuntimeError(f"{cell}: reconstructed SVHN predictions differ from the network's own.")

    ana = retr.analysis
    som = load_selected_som(inputs["som_path"], ana.grid_height, ana.grid_width, ana.feature_dim)
    qe = {k: assign(som, x)[1] for k, x in
          (("train", inputs["train_x"]), ("normal", inputs["val_x"]), ("novel", svhn_x))}
    if not np.allclose(qe["normal"], inputs["val_dist"], rtol=1e-6, atol=1e-9):
        raise RuntimeError(f"{cell}: SOM distances differ from the re-analysis record.")

    train_labels = np.load(retr.embeddings_dir / "train_labels.npy")
    if len(train_labels) != len(inputs["train_x"]):
        raise RuntimeError(f"{cell}: training labels and embeddings differ in length.")
    x = {"train": inputs["train_x"], "normal": inputs["val_x"], "novel": svhn_x}
    scores = {k: score_split(x[k], logits[k], qe[k], inputs["train_x"], train_labels, settings.energy_temperature)
              for k in x}
    metrics = cell_metrics(scores["train"], scores["normal"], scores["novel"], settings.threshold_percentile,
                           settings.bootstrap_resamples, settings.bootstrap_seed, settings.confidence)
    folder = settings.folder("scores")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{cell}.npz", **{f"{k}_{n}": v for k in ("normal", "novel")
                                                    for n, v in scores[k].items()})
    write_json(out, {"cell": cell, "n_normal": int(len(inputs["val_x"])), "n_novel": int(len(svhn_x)),
                     "checkpoint_sha256": sha256_file(inputs["checkpoint"]),
                     "som_model_sha256": sha256_file(inputs["som_path"]), f"{retr.dataset.name}_test_read": False, **metrics})
    log.info("%s: QE AUROC %.4f, max-softmax %.4f, energy %.4f, Mahalanobis %.4f.", cell,
             *(metrics["scores"][n]["auroc"] for n in ("qe", "msp", "energy", "mahalanobis")))


def run_scores(settings: NoveltySettings, only: list[str] | None = None) -> None:
    """Score the reference and every retrained cell (or only the named cells)."""
    start_run(settings)
    device = resolve_device(settings.device)
    for cell in (settings.reference, *settings.cells):
        if only and cell not in only:
            continue
        verify_run(settings)
        score_cell(settings, cell, device)
