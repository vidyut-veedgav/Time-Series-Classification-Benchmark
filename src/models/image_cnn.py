"""Image-based CNN models that treat time series as 2D images."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SignalToImage:
    """
    Transform 1D time series signals to 2D image representations.

    Supports multiple transformation methods:
    - Spectrogram (STFT-based)
    - Gramian Angular Field (GAF)
    - Recurrence Plot
    """

    def __init__(
        self,
        method: str = "spectrogram",
        image_size: int = 64,
        n_fft: int = 64,
        hop_length: int = 8
    ):
        self.method = method
        self.image_size = image_size
        self.n_fft = n_fft
        self.hop_length = hop_length

    def spectrogram(self, x: np.ndarray) -> np.ndarray:
        """Convert signal to spectrogram using short-time Fourier transform."""
        if x.ndim == 1:
            x = x.reshape(-1)

        window = np.hanning(self.n_fft)
        n_frames = (len(x) - self.n_fft) // self.hop_length + 1

        if n_frames < 1:
            x = np.pad(x, (0, self.n_fft - len(x) + self.hop_length))
            n_frames = (len(x) - self.n_fft) // self.hop_length + 1

        spec = np.zeros((self.n_fft // 2 + 1, n_frames))

        for i in range(n_frames):
            start = i * self.hop_length
            frame = x[start:start + self.n_fft] * window
            fft = np.fft.rfft(frame)
            spec[:, i] = np.abs(fft)

        spec = np.log1p(spec)

        spec = self._resize_image(spec, self.image_size, self.image_size)

        return spec

    def gramian_angular_field(self, x: np.ndarray) -> np.ndarray:
        """Convert signal to Gramian Angular Summation Field."""
        x = x.flatten()

        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 0:
            x_scaled = (x - x_min) / (x_max - x_min) * 2 - 1
        else:
            x_scaled = np.zeros_like(x)

        x_scaled = np.clip(x_scaled, -1, 1)

        phi = np.arccos(x_scaled)

        n = len(phi)
        gaf = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gaf[i, j] = np.cos(phi[i] + phi[j])

        gaf = self._resize_image(gaf, self.image_size, self.image_size)

        return gaf

    def recurrence_plot(self, x: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Convert signal to recurrence plot."""
        x = x.flatten()
        n = len(x)

        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.abs(x[i] - x[j])

        if dist_matrix.max() > 0:
            dist_matrix = dist_matrix / dist_matrix.max()

        rp = (dist_matrix < threshold).astype(float)

        rp = self._resize_image(rp, self.image_size, self.image_size)

        return rp

    def _resize_image(self, img: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resize image using bilinear interpolation."""
        h, w = img.shape

        y_ratio = h / height
        x_ratio = w / width

        resized = np.zeros((height, width))

        for i in range(height):
            for j in range(width):
                y = int(i * y_ratio)
                x = int(j * x_ratio)
                y = min(y, h - 1)
                x = min(x, w - 1)
                resized[i, j] = img[y, x]

        return resized

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform signal to 2D image using the configured method."""
        if self.method == "spectrogram":
            return self.spectrogram(x)
        elif self.method == "gaf":
            return self.gramian_angular_field(x)
        elif self.method == "recurrence":
            return self.recurrence_plot(x)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class ImageCNNClassifier(nn.Module):
    """
    2D CNN classifier that treats time series as images.

    Converts 1D signals to 2D representations (spectrogram, GAF, or recurrence plot)
    and uses a standard image classification CNN.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        n_classes: int,
        image_size: int = 64,
        transform_method: str = "spectrogram",
        n_channels: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_length = seq_length
        self.image_size = image_size
        self.transform_method = transform_method

        self.transformer = SignalToImage(
            method=transform_method,
            image_size=image_size
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_features, n_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(n_channels * 4, n_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * 4, n_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_channels * 2, n_classes)
        )

    def signal_to_image(self, x: torch.Tensor) -> torch.Tensor:
        """Convert batch of signals to batch of images."""
        batch_size = x.size(0)
        device = x.device

        x_np = x.cpu().numpy()

        images = np.zeros((batch_size, self.n_features, self.image_size, self.image_size))

        for b in range(batch_size):
            for f in range(self.n_features):
                signal = x_np[b, :, f]
                images[b, f] = self.transformer.transform(signal)

        return torch.FloatTensor(images).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        images = self.signal_to_image(x)

        features = self.conv_layers(images)

        output = self.classifier(features)

        return output


class MultiScaleImageCNN(nn.Module):
    """
    Multi-scale image CNN that uses multiple transform methods.

    Combines features from spectrogram, GAF, and recurrence plot
    for more robust classification.
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        n_classes: int,
        image_size: int = 64,
        n_channels: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_length = seq_length
        self.image_size = image_size

        self.methods = ["spectrogram", "gaf", "recurrence"]
        self.transformers = {
            method: SignalToImage(method=method, image_size=image_size)
            for method in self.methods
        }

        self.branches = nn.ModuleDict()
        for method in self.methods:
            self.branches[method] = nn.Sequential(
                nn.Conv2d(n_features, n_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_channels * 2),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(n_channels * 2, n_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_channels * 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),

                nn.Flatten()
            )

        self.classifier = nn.Sequential(
            nn.Linear(n_channels * 2 * len(self.methods), n_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_channels * 2, n_classes)
        )

    def signal_to_images(self, x: torch.Tensor) -> dict:
        """Convert signals to images using all methods."""
        batch_size = x.size(0)
        device = x.device

        x_np = x.cpu().numpy()

        images = {}
        for method in self.methods:
            method_images = np.zeros((batch_size, self.n_features, self.image_size, self.image_size))

            for b in range(batch_size):
                for f in range(self.n_features):
                    signal = x_np[b, :, f]
                    method_images[b, f] = self.transformers[method].transform(signal)

            images[method] = torch.FloatTensor(method_images).to(device)

        return images

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        images = self.signal_to_images(x)

        features = []
        for method in self.methods:
            feat = self.branches[method](images[method])
            features.append(feat)

        combined = torch.cat(features, dim=-1)

        output = self.classifier(combined)

        return output
