"""VAE models for time series anomaly detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAEAnomalyDetector(nn.Module):
    """
    Variational Autoencoder for time series anomaly detection.

    Reconstructs input sequences and uses reconstruction error for anomaly scoring.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        hidden_size: int = 64,
        latent_dim: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_length = seq_length
        self.latent_dim = latent_dim

        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, hidden_size)

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.output_layer = nn.Linear(hidden_size, n_features)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        _, (h_n, _) = self.encoder_lstm(x)

        h = h_n[-1]

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        batch_size = z.size(0)

        h = self.fc_decode(z)

        decoder_input = h.unsqueeze(1).repeat(1, self.seq_length, 1)

        decoder_out, _ = self.decoder_lstm(decoder_input)

        reconstruction = self.output_layer(decoder_out)

        return reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and latent params."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss with optional beta weighting.

        Returns total loss, reconstruction loss, and KL divergence.
        """
        recon_loss = F.mse_loss(reconstruction, x, reduction="mean")

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores based on reconstruction error.

        Higher scores indicate more anomalous points.
        """
        self.eval()
        with torch.no_grad():
            reconstruction, _, _ = self.forward(x)
            scores = torch.mean((x - reconstruction) ** 2, dim=-1)

        return scores
