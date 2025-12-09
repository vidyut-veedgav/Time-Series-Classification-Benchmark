"""VAE model for time series classification."""

import torch
import torch.nn as nn
from typing import Tuple


class VAEClassifier(nn.Module):
    """
    Variational Autoencoder-based time series classifier.

    Uses latent representations from VAE encoder for classification.
    Architecture inspired by LSTM/RNN classifiers for consistency.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        n_classes: int,
        hidden_size: int = 64,
        latent_dim: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        # Encoder: LSTM → latent distribution
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # Classifier head: latent → classes (similar to LSTM/RNN classifier)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        # Process sequence through LSTM
        _, (h_n, _) = self.encoder_lstm(x)

        # Use final hidden state
        h = h_n[-1]

        # Compute latent distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning class logits.

        Compatible with benchmark_classification which expects logits output.
        """
        # Encode to latent space
        mu, logvar = self.encode(x)

        # Sample latent representation
        z = self.reparameterize(mu, logvar)

        # Classify from latent representation
        logits = self.classifier(z)

        return logits
