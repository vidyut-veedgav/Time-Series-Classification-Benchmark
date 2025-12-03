"""RNN models for time series tasks."""

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """
    Vanilla RNN-based time series classifier.

    Uses PyTorch's built-in RNN layers with optional GRU variant.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        n_classes: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_gru: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        rnn_class = nn.GRU if use_gru else nn.RNN

        self.rnn = rnn_class(
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
        rnn_out, h_n = self.rnn(x)

        if self.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]

        return self.classifier(hidden)


class RNNForecaster(nn.Module):
    """
    RNN-based time series forecaster.

    Simple encoder-decoder architecture using GRU cells.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        pred_length: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_gru: bool = True
    ):
        super().__init__()

        self.pred_length = pred_length
        self.hidden_size = hidden_size
        self.n_features = n_features
        self.n_layers = n_layers

        rnn_class = nn.GRU if use_gru else nn.RNN

        self.encoder = rnn_class(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.decoder = rnn_class(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.output_projection = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        _, h_n = self.encoder(x)

        decoder_input = x[:, -1:, :]

        outputs = []
        hidden = h_n

        for _ in range(self.pred_length):
            decoder_out, hidden = self.decoder(decoder_input, hidden)
            output = self.output_projection(decoder_out)
            outputs.append(output)
            decoder_input = output

        return torch.cat(outputs, dim=1)
