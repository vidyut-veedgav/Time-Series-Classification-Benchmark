"""Transformer models for time series tasks."""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer-based time series classifier.

    Uses PyTorch's built-in TransformerEncoder for the backbone.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_length, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)

        x = self.transformer_encoder(x)

        x = x.mean(dim=1)

        return self.classifier(x)


class TransformerForecaster(nn.Module):
    """
    Transformer-based time series forecaster.

    Encoder-decoder architecture for sequence-to-sequence prediction.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        pred_length: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.pred_length = pred_length
        self.d_model = d_model

        self.encoder_input_projection = nn.Linear(n_features, d_model)
        self.decoder_input_projection = nn.Linear(n_features, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max(seq_length, pred_length) + 100, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        self.output_projection = nn.Linear(d_model, n_features)

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = src.size(0)

        src = self.encoder_input_projection(src)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)

        if tgt is None:
            tgt = torch.zeros(batch_size, self.pred_length, self.d_model, device=src.device)
        else:
            tgt = self.decoder_input_projection(tgt)

        tgt = tgt.permute(1, 0, 2)
        tgt = self.pos_encoder(tgt)
        tgt = tgt.permute(1, 0, 2)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            self.pred_length, device=src.device
        )

        output = self.transformer(src, tgt, tgt_mask=tgt_mask)

        return self.output_projection(output)
