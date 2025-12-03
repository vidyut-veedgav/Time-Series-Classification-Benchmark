"""Plotting utilities for time series experiments."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional
from benchmark import BenchmarkResult


RESULTS_DIR = Path(__file__).parent.parent / "results"


def setup_style():
    """Set up matplotlib style for consistent plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


def plot_training_curves(
    results: List[BenchmarkResult],
    save_path: Optional[Path] = None
) -> None:
    """Plot training loss curves for all models."""
    setup_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    for result in results:
        epochs = [h["epoch"] for h in result.epoch_history]
        train_loss = [h["train_loss"] for h in result.epoch_history]
        ax.plot(epochs, train_loss, label=result.model_name, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Curves")
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_metric_comparison(
    results: List[BenchmarkResult],
    metric: str,
    save_path: Optional[Path] = None
) -> None:
    """Plot bar chart comparing a specific metric across models."""
    setup_style()

    model_names = [r.model_name for r in results]
    values = [r.metrics.get(metric, 0) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("husl", len(model_names))
    bars = ax.bar(model_names, values, color=colors)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_xlabel("Model")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison: {metric.replace('_', ' ').title()}")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_all_metrics(
    results: List[BenchmarkResult],
    save_path: Optional[Path] = None
) -> None:
    """Plot heatmap of all metrics for all models."""
    setup_style()

    model_names = [r.model_name for r in results]

    all_metrics = set()
    for r in results:
        all_metrics.update(r.metrics.keys())
    all_metrics = sorted([m for m in all_metrics if m != "threshold"])

    data = np.zeros((len(model_names), len(all_metrics)))
    for i, result in enumerate(results):
        for j, metric in enumerate(all_metrics):
            data[i, j] = result.metrics.get(metric, 0)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        data,
        annot=True,
        fmt=".4f",
        xticklabels=[m.replace("_", " ").title() for m in all_metrics],
        yticklabels=model_names,
        cmap="YlGnBu",
        ax=ax
    )

    ax.set_title("Model Performance Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_time_comparison(
    results: List[BenchmarkResult],
    save_path: Optional[Path] = None
) -> None:
    """Plot training and inference time comparison."""
    setup_style()

    model_names = [r.model_name for r in results]
    train_times = [r.train_time for r in results]
    inference_times = [r.inference_time for r in results]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, train_times, width, label="Training Time", color="steelblue")
    bars2 = ax.bar(x + width/2, inference_times, width, label="Inference Time", color="coral")

    ax.set_xlabel("Model")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training and Inference Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_parameter_vs_performance(
    results: List[BenchmarkResult],
    metric: str = "accuracy",
    save_path: Optional[Path] = None
) -> None:
    """Plot scatter of model parameters vs performance."""
    setup_style()

    model_names = [r.model_name for r in results]
    n_params = [r.n_parameters for r in results]
    performance = [r.metrics.get(metric, 0) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("husl", len(model_names))
    scatter = ax.scatter(n_params, performance, c=colors, s=200, alpha=0.7)

    for i, name in enumerate(model_names):
        ax.annotate(
            name,
            (n_params[i], performance[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10
        )

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Complexity vs {metric.replace('_', ' ').title()}")

    ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_forecasting_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    n_samples: int = 3,
    save_path: Optional[Path] = None
) -> None:
    """Plot forecasting predictions vs ground truth."""
    setup_style()

    n_samples = min(n_samples, len(predictions))

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        pred_length = predictions.shape[1]

        if predictions.ndim == 3:
            pred = predictions[i, :, -1]
            target = targets[i, :, -1]
        else:
            pred = predictions[i]
            target = targets[i]

        x = np.arange(pred_length)

        ax.plot(x, target, label="Ground Truth", color="steelblue", linewidth=2)
        ax.plot(x, pred, label="Prediction", color="coral", linewidth=2, linestyle="--")

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.set_title(f"{model_name} Forecast (Sample {i+1})")
        ax.legend()

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_anomaly_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    model_name: str,
    save_path: Optional[Path] = None
) -> None:
    """Plot anomaly scores with threshold and ground truth labels."""
    setup_style()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    x = np.arange(len(scores))

    axes[0].plot(x, scores, label="Anomaly Score", color="steelblue", alpha=0.7)
    axes[0].axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.4f})")
    axes[0].set_ylabel("Anomaly Score")
    axes[0].set_title(f"{model_name} Anomaly Detection")
    axes[0].legend()

    anomaly_indices = np.where(labels > 0)[0]
    axes[1].plot(x, labels, label="Ground Truth", color="coral", alpha=0.7)
    if len(anomaly_indices) > 0:
        axes[1].scatter(
            anomaly_indices,
            labels[anomaly_indices],
            color="red",
            s=20,
            label="Anomalies"
        )
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Label")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def create_summary_table(results: List[BenchmarkResult]) -> str:
    """Create a formatted summary table of results."""
    headers = ["Model", "Task", "Dataset", "Parameters", "Train Time", "Key Metric"]
    rows = []

    for r in results:
        if r.task_type == "classification":
            key_metric = f"Accuracy: {r.metrics.get('accuracy', 0):.4f}"
        elif r.task_type == "forecasting":
            key_metric = f"MSE: {r.metrics.get('mse', 0):.4f}"
        else:
            key_metric = f"F1: {r.metrics.get('f1', 0):.4f}"

        rows.append([
            r.model_name,
            r.task_type,
            r.dataset_name,
            f"{r.n_parameters:,}",
            f"{r.train_time:.2f}s",
            key_metric
        ])

    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def format_row(row):
        return "|" + "|".join(f" {str(cell):^{col_widths[i]}} " for i, cell in enumerate(row)) + "|"

    lines = [separator, format_row(headers), separator]
    for row in rows:
        lines.append(format_row(row))
    lines.append(separator)

    return "\n".join(lines)
