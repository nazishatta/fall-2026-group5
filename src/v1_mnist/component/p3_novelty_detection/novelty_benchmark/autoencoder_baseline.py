"""Feature-reconstruction autoencoder for MNIST novelty benchmarking."""

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FeatureAutoencoder(nn.Module):
    """Reconstruct CNN embeddings using a small bottleneck network."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def fit_autoencoder(
    train_features: NDArray[np.float32],
    validation_features: NDArray[np.float32],
    config: dict,
) -> tuple[NDArray[np.float64], list[float]]:
    """Train only on MNIST training embeddings; score validation reconstruction."""
    settings = config["autoencoder"]
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    train = torch.as_tensor(np.asarray(train_features), dtype=torch.float32)
    validation = torch.as_tensor(np.asarray(validation_features), dtype=torch.float32)

    input_dim = int(settings["input_dim"])
    if train.ndim != 2 or validation.ndim != 2:
        raise ValueError("Expected 2D embeddings")
    if train.shape[1] != input_dim or validation.shape[1] != input_dim:
        raise ValueError("Embedding dimensions do not match autoencoder")
    if not torch.isfinite(train).all() or not torch.isfinite(validation).all():
        raise ValueError("Embeddings contain non-finite values")

    network = FeatureAutoencoder(
        input_dim,
        int(settings["hidden_dim"]),
        int(settings["latent_dim"]),
    )
    optimizer = torch.optim.Adam(
        network.parameters(), lr=float(settings["learning_rate"])
    )
    loader = DataLoader(
        TensorDataset(train),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    losses: list[float] = []
    for _ in range(int(settings["epochs"])):
        network.train()
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(network(batch), batch)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        losses.append(total / len(train))

    network.eval()
    with torch.no_grad():
        reconstructed = network(validation)
        scores = torch.mean((reconstructed - validation) ** 2, dim=1)

    return scores.numpy().astype(np.float64), losses
