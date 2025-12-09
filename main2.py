"""
Time Series Anomaly Detection Benchmark Script

Benchmarks 4 different model architectures on anomaly detection:
- LSTM (Long Short-Term Memory)
- RNN (Recurrent Neural Network with GRU)
- CNN (Convolutional Neural Network)
- VAE (Variational Autoencoder)

Each model is tested with multiple hyperparameter configurations.
Results include precision, recall, F1, threshold, train/inference time, and parameters.

Usage:
    python main2.py
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple

from src.data_loader import create_anomaly_loaders
from src.models.lstm import LSTMForecaster
from src.models.rnn import RNNForecaster
from src.models.cnn import CNNForecaster
from src.models.vae import VAEAnomalyDetector
from src.benchmark import (
    benchmark_anomaly_detection,
    BenchmarkResult,
    save_results
)

# Configuration
RESULTS_DIR = Path(__file__).parent / "results" / "anomaly_benchmark"
N_EPOCHS = 20
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Anomaly Detector Wrappers for LSTM, RNN, CNN
# ============================================================================

class LSTMAnomalyDetector(nn.Module):
    """LSTM-based anomaly detector using reconstruction error."""

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Decoder
        decoder_input_size = hidden_size * self.n_directions
        self.decoder = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returns reconstruction and dummy mu/logvar for compatibility."""
        batch_size = x.size(0)

        # Encode
        encoder_out, (h_n, c_n) = self.encoder(x)

        # Use encoder output as decoder input
        decoder_out, _ = self.decoder(encoder_out)
        reconstruction = self.output_layer(decoder_out)

        # Return dummy mu and logvar for compatibility with benchmark code
        mu = torch.zeros(batch_size, 16).to(x.device)
        logvar = torch.zeros(batch_size, 16).to(x.device)

        return reconstruction, mu, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss (no KL for non-VAE)."""
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction="mean")
        kl_loss = torch.tensor(0.0).to(x.device)
        total_loss = recon_loss

        return total_loss, recon_loss, kl_loss

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores based on reconstruction error."""
        self.eval()
        with torch.no_grad():
            reconstruction, _, _ = self.forward(x)
            scores = torch.mean((x - reconstruction) ** 2, dim=-1)

        return scores


class RNNAnomalyDetector(nn.Module):
    """RNN-based anomaly detector using reconstruction error."""

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        # Encoder
        self.encoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Decoder
        decoder_input_size = hidden_size * (2 if bidirectional else 1)
        self.decoder = nn.GRU(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returns reconstruction and dummy mu/logvar."""
        batch_size = x.size(0)

        # Encode
        encoder_out, h_n = self.encoder(x)

        # Use encoder output as decoder input
        decoder_out, _ = self.decoder(encoder_out)
        reconstruction = self.output_layer(decoder_out)

        # Dummy for compatibility
        mu = torch.zeros(batch_size, 16).to(x.device)
        logvar = torch.zeros(batch_size, 16).to(x.device)

        return reconstruction, mu, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss."""
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction="mean")
        kl_loss = torch.tensor(0.0).to(x.device)
        total_loss = recon_loss

        return total_loss, recon_loss, kl_loss

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores based on reconstruction error."""
        self.eval()
        with torch.no_grad():
            reconstruction, _, _ = self.forward(x)
            scores = torch.mean((x - reconstruction) ** 2, dim=-1)

        return scores


class CNNAnomalyDetector(nn.Module):
    """CNN-based anomaly detector using reconstruction error."""

    def __init__(
        self,
        n_features: int,
        seq_length: int,
        n_filters: int = 64,
        kernel_size: int = 3,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_length = seq_length

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Conv1d(n_features, n_filters, 1))

        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation

            encoder_layers.extend([
                nn.Conv1d(n_filters, n_filters, kernel_size, dilation=dilation, padding=padding),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for i in range(n_layers):
            decoder_layers.extend([
                nn.Conv1d(n_filters, n_filters, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        decoder_layers.append(nn.Conv1d(n_filters, n_features, 1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returns reconstruction and dummy mu/logvar."""
        batch_size, seq_len, _ = x.shape

        # Permute for Conv1d
        x_perm = x.permute(0, 2, 1)

        # Encode
        encoded = self.encoder(x_perm)

        # Trim to original length
        encoded = encoded[:, :, :seq_len]

        # Decode
        decoded = self.decoder(encoded)

        # Permute back
        reconstruction = decoded.permute(0, 2, 1)

        # Dummy for compatibility
        mu = torch.zeros(batch_size, 16).to(x.device)
        logvar = torch.zeros(batch_size, 16).to(x.device)

        return reconstruction, mu, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss."""
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction="mean")
        kl_loss = torch.tensor(0.0).to(x.device)
        total_loss = recon_loss

        return total_loss, recon_loss, kl_loss

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores based on reconstruction error."""
        self.eval()
        with torch.no_grad():
            reconstruction, _, _ = self.forward(x)
            scores = torch.mean((x - reconstruction) ** 2, dim=-1)

        return scores


# ============================================================================
# Experiment Functions
# ============================================================================

def run_lstm_experiment() -> List[BenchmarkResult]:
    """Run LSTM anomaly detection experiments with different hyperparameters."""
    print("=" * 80)
    print("Running LSTM Anomaly Detection Experiments")
    print("=" * 80)

    results = []

    # Trial configurations
    configs = [
        {"hidden_size": 64, "n_layers": 2, "bidirectional": True, "window_size": 64},   # baseline
        {"hidden_size": 32, "n_layers": 1, "bidirectional": False, "window_size": 32},  # fast
        {"hidden_size": 128, "n_layers": 2, "bidirectional": True, "window_size": 64},  # large
        {"hidden_size": 64, "n_layers": 1, "bidirectional": True, "window_size": 32},   # efficient
    ]

    for i, config in enumerate(configs, 1):
        print(f"\n--- LSTM Trial {i}/{len(configs)} ---")
        print(f"Config: {config}")

        # Load data with specific window size
        train_loader, val_loader, test_loader, data_info = create_anomaly_loaders(
            window_size=config["window_size"],
            batch_size=BATCH_SIZE
        )

        # Create model
        model = LSTMAnomalyDetector(
            n_features=1,
            seq_length=config["window_size"],
            hidden_size=config["hidden_size"],
            n_layers=config["n_layers"],
            bidirectional=config["bidirectional"]
        )

        # Benchmark
        bidir_str = "bidir" if config["bidirectional"] else "unidir"
        result = benchmark_anomaly_detection(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_name=f"LSTM_h{config['hidden_size']}_l{config['n_layers']}_{bidir_str}_w{config['window_size']}",
            dataset_name="synthetic_anomaly",
            n_epochs=N_EPOCHS,
            device=DEVICE
        )

        results.append(result)
        print(f"F1: {result.metrics['f1']:.4f}, "
              f"Precision: {result.metrics['precision']:.4f}, "
              f"Recall: {result.metrics['recall']:.4f}")

    return results


def run_rnn_experiment() -> List[BenchmarkResult]:
    """Run RNN anomaly detection experiments with different hyperparameters."""
    print("\n" + "=" * 80)
    print("Running RNN Anomaly Detection Experiments")
    print("=" * 80)

    results = []

    # Trial configurations
    configs = [
        {"hidden_size": 64, "n_layers": 2, "bidirectional": True, "window_size": 64},   # baseline
        {"hidden_size": 32, "n_layers": 1, "bidirectional": False, "window_size": 32},  # fast
        {"hidden_size": 128, "n_layers": 2, "bidirectional": True, "window_size": 64},  # large
        {"hidden_size": 64, "n_layers": 1, "bidirectional": True, "window_size": 32},   # efficient
    ]

    for i, config in enumerate(configs, 1):
        print(f"\n--- RNN Trial {i}/{len(configs)} ---")
        print(f"Config: {config}")

        # Load data
        train_loader, val_loader, test_loader, data_info = create_anomaly_loaders(
            window_size=config["window_size"],
            batch_size=BATCH_SIZE
        )

        # Create model
        model = RNNAnomalyDetector(
            n_features=1,
            seq_length=config["window_size"],
            hidden_size=config["hidden_size"],
            n_layers=config["n_layers"],
            bidirectional=config["bidirectional"]
        )

        # Benchmark
        bidir_str = "bidir" if config["bidirectional"] else "unidir"
        result = benchmark_anomaly_detection(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_name=f"RNN_h{config['hidden_size']}_l{config['n_layers']}_{bidir_str}_w{config['window_size']}",
            dataset_name="synthetic_anomaly",
            n_epochs=N_EPOCHS,
            device=DEVICE
        )

        results.append(result)
        print(f"F1: {result.metrics['f1']:.4f}, "
              f"Precision: {result.metrics['precision']:.4f}, "
              f"Recall: {result.metrics['recall']:.4f}")

    return results


def run_cnn_experiment() -> List[BenchmarkResult]:
    """Run CNN anomaly detection experiments with different hyperparameters."""
    print("\n" + "=" * 80)
    print("Running CNN Anomaly Detection Experiments")
    print("=" * 80)

    results = []

    # Trial configurations
    configs = [
        {"n_filters": 64, "n_layers": 4, "kernel_size": 3, "window_size": 64},  # baseline
        {"n_filters": 32, "n_layers": 2, "kernel_size": 3, "window_size": 32},  # fast
        {"n_filters": 128, "n_layers": 4, "kernel_size": 5, "window_size": 64}, # large
    ]

    for i, config in enumerate(configs, 1):
        print(f"\n--- CNN Trial {i}/{len(configs)} ---")
        print(f"Config: {config}")

        # Load data
        train_loader, val_loader, test_loader, data_info = create_anomaly_loaders(
            window_size=config["window_size"],
            batch_size=BATCH_SIZE
        )

        # Create model
        model = CNNAnomalyDetector(
            n_features=1,
            seq_length=config["window_size"],
            n_filters=config["n_filters"],
            kernel_size=config["kernel_size"],
            n_layers=config["n_layers"]
        )

        # Benchmark
        result = benchmark_anomaly_detection(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_name=f"CNN_f{config['n_filters']}_l{config['n_layers']}_k{config['kernel_size']}_w{config['window_size']}",
            dataset_name="synthetic_anomaly",
            n_epochs=N_EPOCHS,
            device=DEVICE
        )

        results.append(result)
        print(f"F1: {result.metrics['f1']:.4f}, "
              f"Precision: {result.metrics['precision']:.4f}, "
              f"Recall: {result.metrics['recall']:.4f}")

    return results


def run_vae_experiment() -> List[BenchmarkResult]:
    """Run VAE anomaly detection experiments with different hyperparameters."""
    print("\n" + "=" * 80)
    print("Running VAE Anomaly Detection Experiments")
    print("=" * 80)

    results = []

    # Trial configurations
    configs = [
        {"hidden_size": 64, "latent_dim": 16, "n_layers": 2, "window_size": 64},  # baseline
        {"hidden_size": 32, "latent_dim": 8, "n_layers": 1, "window_size": 32},   # fast
        {"hidden_size": 128, "latent_dim": 32, "n_layers": 2, "window_size": 64}, # large
        {"hidden_size": 64, "latent_dim": 16, "n_layers": 1, "window_size": 32},  # efficient
    ]

    for i, config in enumerate(configs, 1):
        print(f"\n--- VAE Trial {i}/{len(configs)} ---")
        print(f"Config: {config}")

        # Load data
        train_loader, val_loader, test_loader, data_info = create_anomaly_loaders(
            window_size=config["window_size"],
            batch_size=BATCH_SIZE
        )

        # Create model
        model = VAEAnomalyDetector(
            n_features=1,
            seq_length=config["window_size"],
            hidden_size=config["hidden_size"],
            latent_dim=config["latent_dim"],
            n_layers=config["n_layers"]
        )

        # Benchmark
        result = benchmark_anomaly_detection(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_name=f"VAE_h{config['hidden_size']}_z{config['latent_dim']}_l{config['n_layers']}_w{config['window_size']}",
            dataset_name="synthetic_anomaly",
            n_epochs=N_EPOCHS,
            device=DEVICE
        )

        results.append(result)
        print(f"F1: {result.metrics['f1']:.4f}, "
              f"Precision: {result.metrics['precision']:.4f}, "
              f"Recall: {result.metrics['recall']:.4f}")

    return results


# ============================================================================
# Results Analysis and Visualization
# ============================================================================

def create_results_table(results: List[BenchmarkResult]) -> pd.DataFrame:
    """Create a summary table of all results."""
    data = []

    for result in results:
        row = {
            "Model": result.model_name,
            "Precision": result.metrics["precision"],
            "Recall": result.metrics["recall"],
            "F1 Score": result.metrics["f1"],
            "Threshold": result.metrics["threshold"],
            "Train Time (s)": result.train_time,
            "Inference Time (s)": result.inference_time,
            "Parameters": result.n_parameters
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by F1 score descending
    df = df.sort_values("F1 Score", ascending=False)

    return df


def plot_loss_curves(results: List[BenchmarkResult], filepath: Path):
    """Plot training loss curves for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Group by model type
    model_groups = {}
    for result in results:
        model_type = result.model_name.split("_")[0]
        if model_type not in model_groups:
            model_groups[model_type] = []
        model_groups[model_type].append(result)

    for idx, (model_type, group_results) in enumerate(model_groups.items()):
        ax = axes[idx]

        for result in group_results:
            epochs = [h["epoch"] for h in result.epoch_history]
            train_loss = [h["train_loss"] for h in result.epoch_history]

            ax.plot(epochs, train_loss, label=result.model_name, marker='o', markersize=3)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.set_title(f"{model_type} Training Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Loss curves saved to {filepath}")
    plt.close()


def plot_time_comparison(results: List[BenchmarkResult], filepath: Path):
    """Plot train and inference time comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    model_names = [r.model_name for r in results]
    train_times = [r.train_time for r in results]
    inference_times = [r.inference_time for r in results]

    # Train time
    ax1.barh(model_names, train_times, color='steelblue')
    ax1.set_xlabel("Time (seconds)")
    ax1.set_title("Training Time Comparison")
    ax1.grid(True, alpha=0.3, axis='x')

    # Inference time
    ax2.barh(model_names, inference_times, color='coral')
    ax2.set_xlabel("Time (seconds)")
    ax2.set_title("Inference Time Comparison")
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Time comparison saved to {filepath}")
    plt.close()


def plot_metrics_comparison(results: List[BenchmarkResult], filepath: Path):
    """Plot precision, recall, and F1 comparison."""
    model_names = [r.model_name for r in results]
    precision = [r.metrics["precision"] for r in results]
    recall = [r.metrics["recall"] for r in results]
    f1 = [r.metrics["f1"] for r in results]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    ax.bar(x, recall, width, label='Recall', color='coral')
    ax.bar(x + width, f1, width, label='F1 Score', color='seagreen')

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Anomaly Detection Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison saved to {filepath}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run anomaly detection benchmark experiments")
    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm", "rnn", "cnn", "vae", "all"],
        default="all",
        help="Which model(s) to benchmark"
    )

    args = parser.parse_args()

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")

    # Run experiments
    all_results = []

    if args.model in ["lstm", "all"]:
        lstm_results = run_lstm_experiment()
        all_results.extend(lstm_results)

    if args.model in ["rnn", "all"]:
        rnn_results = run_rnn_experiment()
        all_results.extend(rnn_results)

    if args.model in ["cnn", "all"]:
        cnn_results = run_cnn_experiment()
        all_results.extend(cnn_results)

    if args.model in ["vae", "all"]:
        vae_results = run_vae_experiment()
        all_results.extend(vae_results)

    # Save results
    save_results(all_results, RESULTS_DIR / "benchmark_results.json")

    # Create summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    results_df = create_results_table(all_results)
    print(results_df.to_string(index=False))

    # Save table to CSV
    results_df.to_csv(RESULTS_DIR / "results_table.csv", index=False)
    print(f"\nResults table saved to {RESULTS_DIR / 'results_table.csv'}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_loss_curves(all_results, RESULTS_DIR / "loss_curves.png")
    plot_time_comparison(all_results, RESULTS_DIR / "time_comparison.png")
    plot_metrics_comparison(all_results, RESULTS_DIR / "metrics_comparison.png")

    print("\n" + "=" * 80)
    print("All experiments completed successfully!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
