import torch

from src.v1_mnist.component.models.lenet5 import LeNet5


def test_forward_shape():
    model = LeNet5(num_classes=10).eval()
    with torch.no_grad():
        output = model(torch.randn(4, 1, 28, 28))
    assert output.shape == (4, 10)


def test_feature_layer():
    model = LeNet5(num_classes=10)
    assert model.feature_layer_name == "fc2"
    assert model.feature_dim == 84
    assert model.fc2.weight.shape == (84, 120)


def test_output_is_finite():
    model = LeNet5(num_classes=10).eval()
    with torch.no_grad():
        output = model(torch.randn(4, 1, 28, 28))
    assert torch.isfinite(output).all()
