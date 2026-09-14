import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.component.features.extractor import FeatureExtractor
from src.component.models.lenet5 import LeNet5


def make_loader(n=20):
    images = torch.randn(n, 1, 28, 28)
    labels = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=10)


def test_embedding_shape():
    extractor = FeatureExtractor(LeNet5(num_classes=10).eval(), layer_name="fc2")
    features, labels, preds = extractor.extract(
        make_loader(), device=torch.device("cpu")
    )
    assert features.shape == (20, 84)
    assert len(labels) == len(preds) == 20


def test_embeddings_are_finite():
    extractor = FeatureExtractor(LeNet5(num_classes=10).eval(), layer_name="fc2")
    features, _, _ = extractor.extract(
        make_loader(), device=torch.device("cpu")
    )
    assert np.isfinite(features).all()


def test_metadata_output():
    extractor = FeatureExtractor(LeNet5(num_classes=10).eval(), layer_name="fc2")
    features, labels, preds, ids, confidence, correct = extractor.extract(
        make_loader(),
        device=torch.device("cpu"),
        return_metadata=True,
    )
    assert len(features) == len(labels) == len(preds) == len(confidence) == len(correct) == 20
    assert ((confidence >= 0) & (confidence <= 1)).all()
    assert correct.dtype == bool
