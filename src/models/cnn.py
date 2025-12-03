"""CNN models for time series tasks."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)

        return out


class CNNClassifier(nn.Module):
    """
    CNN-based time series classifier.

    Uses 1D convolutions with residual connections.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        n_classes: int,
        n_filters: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_conv = nn.Conv1d(n_features, n_filters, 1)

        blocks = []
        for i in range(n_blocks):
            in_ch = n_filters * (2 ** i) if i > 0 else n_filters
            out_ch = n_filters * (2 ** i)
            blocks.append(ConvBlock(in_ch, out_ch, kernel_size, dropout))

            if i < n_blocks - 1:
                blocks.append(nn.Conv1d(out_ch, out_ch * 2, 1))

        self.conv_blocks = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        final_channels = n_filters * (2 ** (n_blocks - 1))
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, n_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_filters, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        x = self.input_conv(x)
        x = self.conv_blocks(x)

        x = self.global_pool(x).squeeze(-1)

        return self.classifier(x)


class CNNForecaster(nn.Module):
    """
    CNN-based time series forecaster.

    Uses dilated causal convolutions for sequence modeling.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        pred_length: int,
        n_filters: int = 64,
        kernel_size: int = 3,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.pred_length = pred_length
        self.n_features = n_features

        self.input_conv = nn.Conv1d(n_features, n_filters, 1)

        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation

            layers.extend([
                nn.Conv1d(
                    n_filters, n_filters, kernel_size,
                    dilation=dilation, padding=padding
                ),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        self.temporal_conv = nn.Sequential(*layers)

        self.output_conv = nn.Conv1d(n_filters, n_features, 1)

        self.fc = nn.Linear(seq_length, pred_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x = x.permute(0, 2, 1)

        x = self.input_conv(x)
        x = self.temporal_conv(x)

        x = x[:, :, :seq_len]

        x = self.output_conv(x)

        x = x.permute(0, 2, 1)

        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)

        return x
