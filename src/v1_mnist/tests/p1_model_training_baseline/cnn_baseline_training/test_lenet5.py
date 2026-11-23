import pytest
import torch

from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.lenet5 import LeNet5


def test_forward_shape() -> None:
    model = LeNet5(num_classes=10).eval()
    with torch.no_grad():
        output = model(torch.randn(4, 1, 28, 28))
    assert output.shape == (4, 10)


def test_feature_layer() -> None:
    model = LeNet5(num_classes=10)
    assert model.feature_layer_name == "fc2"
    assert model.feature_dim == 84
    assert model.fc2.weight.shape == (84, 120)


def test_output_is_finite() -> None:
    model = LeNet5(num_classes=10).eval()
    with torch.no_grad():
        output = model(torch.randn(4, 1, 28, 28))
    assert torch.isfinite(output).all()


def test_default_input_keeps_the_original_layer_sizes() -> None:
    """MNIST checkpoints must still load: 1 input channel and a 16 x 5 x 5 flattened size."""
    model = LeNet5(num_classes=10)
    assert model.conv1.in_channels == 1
    assert model.fc1.in_features == 16 * 5 * 5


def test_input_channels_and_size_come_from_the_arguments() -> None:
    model = LeNet5(num_classes=10, in_channels=3, image_size=32).eval()
    assert model.fc1.in_features == 16 * 6 * 6
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_too_small_images_are_refused() -> None:
    with pytest.raises(ValueError, match="too small"):
        LeNet5(image_size=8)
