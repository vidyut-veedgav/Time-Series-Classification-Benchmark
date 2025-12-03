"""STL decomposition based models for time series tasks."""

import torch
import torch.nn as nn
import numpy as np
from scipy import signal
from typing import Tuple, Optional


class STLDecomposer:
    """
    STL-like decomposition for time series.

    Decomposes signal into trend, seasonal, and residual components
    using moving averages and filtering.
    """

    def __init__(self, period: int = 24, trend_window: int = 25):
        self.period = period
        self.trend_window = trend_window if trend_window % 2 == 1 else trend_window + 1

    def decompose(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose time series into trend, seasonal, and residual.

        Args:
            x: Input array of shape (seq_length,) or (seq_length, n_features)

        Returns:
            Tuple of (trend, seasonal, residual) arrays
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n_samples, n_features = x.shape
        trend = np.zeros_like(x)
        seasonal = np.zeros_like(x)
        residual = np.zeros_like(x)

        for f in range(n_features):
            series = x[:, f]

            kernel = np.ones(self.trend_window) / self.trend_window
            trend_component = np.convolve(series, kernel, mode="same")

            detrended = series - trend_component

            seasonal_component = np.zeros_like(series)
            for i in range(self.period):
                indices = np.arange(i, len(series), self.period)
                if len(indices) > 0:
                    seasonal_mean = np.mean(detrended[indices])
                    seasonal_component[indices] = seasonal_mean

            residual_component = series - trend_component - seasonal_component

            trend[:, f] = trend_component
            seasonal[:, f] = seasonal_component
            residual[:, f] = residual_component

        return trend, seasonal, residual


class STLForecaster(nn.Module):
    """
    STL-based forecaster that models each component separately.

    Uses neural networks to forecast trend, seasonal, and residual
    components independently, then combines them.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        pred_length: int,
        hidden_size: int = 64,
        period: int = 24,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.period = period

        self.decomposer = STLDecomposer(period=period)

        self.trend_model = nn.Sequential(
            nn.Linear(seq_length * n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pred_length * n_features)
        )

        self.seasonal_model = nn.Sequential(
            nn.Linear(seq_length * n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_length * n_features)
        )

        self.residual_model = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.residual_fc = nn.Linear(hidden_size, n_features)

    def decompose_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose a batch of sequences."""
        batch_size = x.size(0)
        device = x.device

        x_np = x.cpu().numpy()

        trends = []
        seasonals = []
        residuals = []

        for i in range(batch_size):
            trend, seasonal, residual = self.decomposer.decompose(x_np[i])
            trends.append(trend)
            seasonals.append(seasonal)
            residuals.append(residual)

        trend_t = torch.FloatTensor(np.array(trends)).to(device)
        seasonal_t = torch.FloatTensor(np.array(seasonals)).to(device)
        residual_t = torch.FloatTensor(np.array(residuals)).to(device)

        return trend_t, seasonal_t, residual_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        trend, seasonal, residual = self.decompose_batch(x)

        trend_flat = trend.reshape(batch_size, -1)
        trend_pred = self.trend_model(trend_flat)
        trend_pred = trend_pred.reshape(batch_size, self.pred_length, self.n_features)

        seasonal_flat = seasonal.reshape(batch_size, -1)
        seasonal_pred = self.seasonal_model(seasonal_flat)
        seasonal_pred = seasonal_pred.reshape(batch_size, self.pred_length, self.n_features)

        residual_out, _ = self.residual_model(residual)
        residual_pred = self.residual_fc(residual_out[:, -self.pred_length:, :])

        if residual_pred.size(1) < self.pred_length:
            pad_size = self.pred_length - residual_pred.size(1)
            residual_pred = torch.cat([
                residual_pred,
                residual_pred[:, -1:, :].repeat(1, pad_size, 1)
            ], dim=1)

        output = trend_pred + seasonal_pred + residual_pred

        return output


class STLClassifier(nn.Module):
    """
    STL-based classifier that uses decomposed components as features.

    Extracts features from trend, seasonal, and residual components
    for classification.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        n_classes: int,
        hidden_size: int = 64,
        period: int = 24,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_length = seq_length
        self.period = period

        self.decomposer = STLDecomposer(period=period)

        input_size = seq_length * n_features * 3

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

        self.component_encoder = nn.ModuleList([
            nn.Linear(seq_length * n_features, hidden_size)
            for _ in range(3)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def decompose_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose a batch of sequences."""
        batch_size = x.size(0)
        device = x.device

        x_np = x.cpu().numpy()

        trends = []
        seasonals = []
        residuals = []

        for i in range(batch_size):
            trend, seasonal, residual = self.decomposer.decompose(x_np[i])
            trends.append(trend)
            seasonals.append(seasonal)
            residuals.append(residual)

        trend_t = torch.FloatTensor(np.array(trends)).to(device)
        seasonal_t = torch.FloatTensor(np.array(seasonals)).to(device)
        residual_t = torch.FloatTensor(np.array(residuals)).to(device)

        return trend_t, seasonal_t, residual_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        trend, seasonal, residual = self.decompose_batch(x)

        components = [trend, seasonal, residual]
        encoded = []

        for i, comp in enumerate(components):
            comp_flat = comp.reshape(batch_size, -1)
            enc = self.component_encoder[i](comp_flat)
            encoded.append(enc)

        fused = torch.cat(encoded, dim=-1)
        output = self.fusion(fused)

        return output
