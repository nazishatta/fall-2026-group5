"""Reproduce frozen QE and MSP Fashion-MNIST evaluation without writing files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_metrics import (
    ood_metrics,
    stratified_auroc_interval,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_scoring import (
    score_embeddings,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection.run_qe_ood import (
    prepare,
    read_settings,
)

LOGGER = logging.getLogger(__name__)
CONFIG = Path("src/v1_mnist/component/configs/qe_ood_evaluation.yaml")
EVIDENCE = Path(
    "outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/"
    "mnist_fashion_qe_msp_s2026_initial/fashion_comparison_metrics.json"
)
EMBEDDINGS = Path(
    "outputs/v1_mnist/p1_model_training_baseline/"
    "cnn_baseline_training/embeddings"
)


def reproduce() -> dict[str, object]:
    """Reproduce both AUROC estimates on identical frozen evaluation samples."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import FashionMNIST

    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (
        FeatureExtractor,
    )
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
        image_transform,
    )

    config = read_settings(CONFIG)
    model, som, validation_qe, provenance = prepare(config)

    torch.set_num_threads(1)
    dataset = FashionMNIST(
        root="data/ood_fashion_mnist",
        train=False,
        download=False,
        transform=image_transform("fashion_mnist", (28, 28)),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    extractor = FeatureExtractor(model, "fc2")
    features, _, _, _, confidence, _ = extractor.extract(
        loader, torch.device("cpu"), return_metadata=True
    )

    validation_confidence = np.load(
        EMBEDDINGS / "val_confidence.npy", allow_pickle=False
    ).reshape(-1)
    confidence = np.asarray(confidence).reshape(-1)

    assert validation_qe.shape == (10500,)
    assert validation_confidence.shape == (10500,)
    assert features.shape == (10000, 84)
    assert confidence.shape == (10000,)
    assert np.isfinite(confidence).all()

    fashion_qe = score_embeddings(som, features)
    qe_metrics = ood_metrics(
        validation_qe,
        fashion_qe,
        float(provenance["qe_threshold"]),
        0.95,
    )

    qe_ci = stratified_auroc_interval(
        validation_qe,
        fashion_qe,
        iterations=2000,
        seed=2026,
        confidence=0.95,
    )

    id_msp = 1.0 - validation_confidence
    ood_msp = 1.0 - confidence

    labels = np.concatenate(
        [np.zeros(len(id_msp), dtype=int), np.ones(len(ood_msp), dtype=int)]
    )
    scores = np.concatenate([id_msp, ood_msp])

    msp_auroc = float(roc_auc_score(labels, scores))
    msp_aupr = float(average_precision_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    msp_fpr95 = float(np.interp(0.95, tpr, fpr))

    msp_ci = stratified_auroc_interval(
        id_msp,
        ood_msp,
        iterations=2000,
        seed=2026,
        confidence=0.95,
    )

    results: dict[str, object] = {
        "quantization_error": {
            "auroc": float(qe_metrics["auroc"]),
            "auroc_ci95": list(qe_ci),
            "aupr_ood": float(qe_metrics["aupr_ood"]),
            "fpr_at_95_tpr": float(qe_metrics["fpr_at_target_tpr"]),
        },
        "max_softmax": {
            "auroc": msp_auroc,
            "auroc_ci95": list(msp_ci),
            "aupr_ood": msp_aupr,
            "fpr_at_95_tpr": msp_fpr95,
        },
    }

    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    for method, metrics in results.items():
        for metric, actual in metrics.items():
            expected = evidence[method][metric]
            if not np.allclose(actual, expected, atol=1e-10, rtol=0):
                raise RuntimeError(
                    f"Reproduction mismatch: {method}.{metric}: "
                    f"actual={actual}, expected={expected}"
                )

    LOGGER.info("PASS: QE and MSP independently reproduce saved evidence")
    return results


def main() -> None:
    """Run a read-only scientific reproduction check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", required=True)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    results = reproduce()
    LOGGER.info("Verified metrics:\n%s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
