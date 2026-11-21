"""Stage 1: one split end to end (protocol sections 3 to 6). Needs PyTorch and NNSOM.

Train LeNet-5 on the known digits, read the fc2 embeddings, train the SOM on the training embeddings,
score the normal (known) and novel (unknown) validation images, and write the split's records.
``metrics.json`` is written last, so a split counts as done only when it exists.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import FeatureExtractor
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.trainer_history import (
    train_som_with_history,
)
from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.class_split import (
    ClassSplit,
    cells_covering,
    make_split,
)
from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.provenance import start_run, verify_run
from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.settings import HeldOutSettings
from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.split_metrics import (
    SCORES,
    all_scores,
    per_digit_qe_auroc,
    split_metrics,
)
from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.scoring import relu_logits
from src.v1_mnist.component.utils.cnn_retraining import load_model, resolve_device, train_cnn
from src.v1_mnist.component.utils.frozen_split import SplitImages, load_train_val_images
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.som_io import reconstruct_assignments, sha256_file

log = logging.getLogger("v1_mnist.held_out_class_novelty")
PARTS = ("train", "normal", "novel")


def _embed(model: torch.nn.Module, images: np.ndarray, layer: str, batch_size: int,
           device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    data = TensorDataset(torch.as_tensor(images).unsqueeze(1), torch.zeros(len(images), dtype=torch.long))
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    features, _labels, predictions = FeatureExtractor(model, layer).extract(loader, device)
    return features.astype(np.float32), np.asarray(predictions, dtype=np.int64)


def _assign(som: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-matching neuron and its distance for every row (the SOM scales the rows itself)."""
    clusters, distances, _max_dist, _sizes = som.cluster_data(x)
    return reconstruct_assignments(clusters, distances, x.shape[0])


def _train_network(settings: HeldOutSettings, cs: ClassSplit, device: torch.device) -> tuple[Any, dict[str, Any]]:
    rows = np.arange(len(cs.train_y))
    result = train_cnn(cs.train_x, cs.train_y, cs.normal_x, cs.normal_y, None, lambda _epoch: rows,
                       settings.train, settings.training_seed, device)
    model = load_model(result.best_state, settings.train.num_classes, device)
    return model, {"best_epoch": result.best_epoch, "best_val_accuracy_known": result.best_val_accuracy,
                   "history": result.history, "state": result.best_state}


def _train_map(settings: HeldOutSettings, train_x: np.ndarray) -> Any:
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(train_x)   # training embeddings only
    r = settings.som
    som, _history = train_som_with_history(
        x_train=train_x, grid_height=r.grid_height, grid_width=r.grid_width,
        init_neighborhood=r.init_neighborhood, epochs=r.epochs, steps=r.steps,
        norm_func=scaler.transform, seed=r.seed,
    )
    return som


def run_split(settings: HeldOutSettings, split: int, train: SplitImages, val: SplitImages,
              device: torch.device) -> None:
    """Train, score and record one split, or skip it if its metrics exist."""
    if settings.metrics_path(split).is_file():
        log.info("split %d: already done, skipped.", split)
        return
    unknown = settings.unknown_digits[split - 1]
    cs = make_split(unknown, settings.num_classes, train, val)
    log.info("split %d: known %s, unknown %s; %d training, %d normal, %d novel images.", split, cs.known,
             cs.unknown, len(cs.train_y), len(cs.normal_y), len(cs.novel_digits))
    folder = settings.split_dir(split)
    folder.mkdir(parents=True, exist_ok=True)

    model, net = _train_network(settings, cs, device)
    torch.save(net.pop("state"), folder / "checkpoint.pth")
    images = {"train": cs.train_x, "normal": cs.normal_x, "novel": cs.novel_x}
    feats, preds = {}, {}
    for part in PARTS:
        feats[part], preds[part] = _embed(model, images[part], settings.layer, settings.batch_size, device)
        np.save(folder / f"{part}_features.npy", feats[part])
    w, b = model.fc3.weight.detach().cpu().numpy(), model.fc3.bias.detach().cpu().numpy()
    logits = {p: relu_logits(feats[p], w, b) for p in PARTS}
    for p in PARTS:
        if not np.array_equal(logits[p].argmax(1), preds[p]):
            raise RuntimeError(f"split {split}: logits rebuilt from fc2 disagree with the network on '{p}'.")

    som = _train_map(settings, feats["train"])
    som.save_pickle("som", str(folder) + os.sep)
    bmu, qe = {}, {}
    for p in PARTS:
        bmu[p], qe[p] = _assign(som, feats[p])

    scores = all_scores(feats, logits, qe, cs.train_y, settings.energy_temperature, settings.knn_neighbors)
    metrics = split_metrics(scores, settings.threshold_percentile, settings.bootstrap_resamples,
                            settings.bootstrap_seed, settings.confidence, settings.auroc_margin,
                            settings.null_iterations, settings.null_seed)
    np.savez_compressed(folder / "scores.npz", **{f"{k}_{n}": v for k in ("normal", "novel")
                                                  for n, v in scores[k].items()},
                        normal_bmu=bmu["normal"], novel_bmu=bmu["novel"], novel_digits=cs.novel_digits,
                        normal_ids=cs.normal_ids, novel_ids=cs.novel_ids)
    write_json(settings.metrics_path(split), {
        "split": split, "known_digits": list(cs.known), "unknown_digits": list(cs.unknown),
        "n_train": int(len(cs.train_y)), "n_normal": int(len(cs.normal_y)), "n_novel": int(len(cs.novel_digits)),
        "closed_set_accuracy_known": float(np.mean(preds["normal"] == cs.normal_y)),
        "network": net, "scores_reported": list(SCORES),
        "per_unknown_digit_qe_auroc": per_digit_qe_auroc(qe["normal"], qe["novel"], cs.novel_digits),
        "cells_holding_share": {"share": settings.coverage_share,
                                "novel": cells_covering(bmu["novel"], settings.coverage_share),
                                "normal": cells_covering(bmu["normal"], settings.coverage_share)},
        "checkpoint_sha256": sha256_file(folder / "checkpoint.pth"),
        "som_model_sha256": sha256_file(folder / "som"),
        "features_sha256": {p: sha256_file(folder / f"{p}_features.npy") for p in PARTS},
        "mnist_test_read": False, **metrics,
    })
    m = metrics["scores"]
    log.info("split %d: AUROC QE %.4f, max-softmax %.4f, energy %.4f, Mahalanobis %.4f, kNN %.4f.", split,
             *(m[n]["auroc"] for n in SCORES))


def run_splits(settings: HeldOutSettings, only: list[int] | None = None) -> None:
    """Run every split (or only the named ones). The MNIST test split is not loaded."""
    start_run(settings)
    train, val = load_train_val_images(settings.data_root, settings.embeddings_dir, settings.dataset)
    device = resolve_device(settings.device)
    for split in settings.split_ids:
        if only and split not in only:
            continue
        verify_run(settings)
        run_split(settings, split, train, val, device)
