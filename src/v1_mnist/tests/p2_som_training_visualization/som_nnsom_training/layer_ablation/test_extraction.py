"""Feature-layer ablation: both layers in one pass, and the fc2 check against Phase 1 (skipped without PyTorch)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (  # noqa: E402
    FeatureExtractor,
)
from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.lenet5 import LeNet5  # noqa: E402
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.extraction import (  # noqa: E402
    check_reference,
    extract_layers,
)


def test_both_layers_match_the_phase1_extractor() -> None:
    torch.manual_seed(0)
    model = LeNet5().eval()
    images = torch.rand(12, 1, 28, 28)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images), batch_size=5)
    feats, preds = extract_layers(model, loader, ("fc1", "fc2"), torch.device("cpu"))
    assert feats["fc1"].shape == (12, 120) and feats["fc2"].shape == (12, 84)
    labelled = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, torch.zeros(12, dtype=torch.long)), batch_size=5)
    reference, _labels, ref_preds = FeatureExtractor(model, "fc2").extract(labelled, torch.device("cpu"))
    assert np.allclose(feats["fc2"], reference, atol=1e-6)
    assert np.array_equal(preds, ref_preds)


def test_reference_check_stops_on_a_difference() -> None:
    x = np.zeros((4, 84), dtype=np.float32)
    p = np.arange(4)
    assert check_reference(x, p, x, p, 1e-4, "val") == 0.0
    with pytest.raises(RuntimeError, match="differs from Phase 1 by"):
        check_reference(x + 1e-3, p, x, p, 1e-4, "val")
    with pytest.raises(RuntimeError, match="predictions differ"):
        check_reference(x, p[::-1], x, p, 1e-4, "val")
