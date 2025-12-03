"""LSTM models for time series tasks."""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM-based time series classifier.

    Uses PyTorch's built-in LSTM layers.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        n_classes: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.n_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]

        return self.classifier(hidden)


class LSTMForecaster(nn.Module):
    """
    LSTM-based time series forecaster.

    Encoder-decoder architecture with attention mechanism.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        pred_length: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.pred_length = pred_length
        self.hidden_size = hidden_size
        self.n_features = n_features

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.decoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.output_projection = nn.Linear(hidden_size * 2, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        encoder_out, (h_n, c_n) = self.encoder(x)

        decoder_input = x[:, -1:, :].repeat(1, self.pred_length, 1)

        decoder_out, _ = self.decoder(decoder_input, (h_n, c_n))

        attn_out, _ = self.attention(decoder_out, encoder_out, encoder_out)

        combined = torch.cat([decoder_out, attn_out], dim=-1)

        return self.output_projection(combined)
