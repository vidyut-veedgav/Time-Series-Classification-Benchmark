"""Benchmarking utilities for time series models."""

import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    task_type: str
    dataset_name: str
    train_time: float
    inference_time: float
    n_parameters: int
    metrics: Dict[str, float]
    epoch_history: List[Dict[str, float]]


class Trainer:
    """Generic trainer for time series models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.history = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            if hasattr(self.model, "loss_function"):
                reconstruction, mu, logvar = self.model(x)
                loss, _, _ = self.model.loss_function(x, reconstruction, mu, logvar)
            else:
                output = self.model(x)
                loss = self.loss_fn(output, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                if hasattr(self.model, "loss_function"):
                    reconstruction, mu, logvar = self.model(x)
                    loss, _, _ = self.model.loss_function(x, reconstruction, mu, logvar)
                    all_preds.append(reconstruction.cpu().numpy())
                    all_targets.append(x.cpu().numpy())
                else:
                    output = self.model(x)
                    loss = self.loss_fn(output, y)

                    if output.dim() == 2 and y.dim() == 1:
                        preds = output.argmax(dim=1).cpu().numpy()
                    else:
                        preds = output.cpu().numpy()

                    all_preds.append(preds)
                    all_targets.append(y.cpu().numpy())

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        return {"loss": avg_loss, "preds": all_preds, "targets": all_targets}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        """Train the model."""
        best_val_loss = float("inf")
        patience_counter = 0

        iterator = range(n_epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training")

        for epoch in iterator:
            train_loss = self.train_epoch(train_loader)

            epoch_metrics = {"epoch": epoch, "train_loss": train_loss}

            if val_loader is not None:
                val_results = self.evaluate(val_loader)
                epoch_metrics["val_loss"] = val_results["loss"]

                if val_results["loss"] < best_val_loss:
                    best_val_loss = val_results["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            self.history.append(epoch_metrics)

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(epoch_metrics)

        return self.history


def compute_classification_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "f1_macro": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "precision": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(targets, preds, average="macro", zero_division=0))
    }


def compute_forecasting_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute forecasting metrics."""
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(mse))

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def compute_anomaly_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold_percentile: float = 95
) -> Dict[str, float]:
    """Compute anomaly detection metrics."""
    threshold = np.percentile(scores, threshold_percentile)
    preds = (scores > threshold).astype(int)

    if labels.ndim > 1:
        labels = labels.flatten()
    if preds.ndim > 1:
        preds = preds.flatten()

    labels = labels[:len(preds)]

    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "threshold": float(threshold)
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_classification(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    dataset_name: str,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> BenchmarkResult:
    """Benchmark a classification model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss_fn, device)

    start_time = time.time()
    history = trainer.fit(train_loader, n_epochs=n_epochs, verbose=True)
    train_time = time.time() - start_time

    start_time = time.time()
    results = trainer.evaluate(test_loader)
    inference_time = time.time() - start_time

    metrics = compute_classification_metrics(results["preds"], results["targets"])
    metrics["test_loss"] = results["loss"]

    return BenchmarkResult(
        model_name=model_name,
        task_type="classification",
        dataset_name=dataset_name,
        train_time=train_time,
        inference_time=inference_time,
        n_parameters=count_parameters(model),
        metrics=metrics,
        epoch_history=history
    )


def benchmark_forecasting(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    dataset_name: str,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> BenchmarkResult:
    """Benchmark a forecasting model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    trainer = Trainer(model, optimizer, loss_fn, device)

    start_time = time.time()
    history = trainer.fit(train_loader, val_loader, n_epochs=n_epochs, verbose=True)
    train_time = time.time() - start_time

    start_time = time.time()
    results = trainer.evaluate(test_loader)
    inference_time = time.time() - start_time

    metrics = compute_forecasting_metrics(results["preds"], results["targets"])
    metrics["test_loss"] = results["loss"]

    return BenchmarkResult(
        model_name=model_name,
        task_type="forecasting",
        dataset_name=dataset_name,
        train_time=train_time,
        inference_time=inference_time,
        n_parameters=count_parameters(model),
        metrics=metrics,
        epoch_history=history
    )


def benchmark_anomaly_detection(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    dataset_name: str,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> BenchmarkResult:
    """Benchmark an anomaly detection model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, optimizer, None, device)

    start_time = time.time()
    history = trainer.fit(train_loader, val_loader, n_epochs=n_epochs, verbose=True)
    train_time = time.time() - start_time

    model.eval()
    all_scores = []
    all_labels = []

    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)

            scores = model.anomaly_score(x)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(y.numpy())

    inference_time = time.time() - start_time

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_anomaly_metrics(all_scores, all_labels)

    return BenchmarkResult(
        model_name=model_name,
        task_type="anomaly_detection",
        dataset_name=dataset_name,
        train_time=train_time,
        inference_time=inference_time,
        n_parameters=count_parameters(model),
        metrics=metrics,
        epoch_history=history
    )


def save_results(results: List[BenchmarkResult], filepath: Path) -> None:
    """Save benchmark results to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = [asdict(r) for r in results]

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {filepath}")


def load_results(filepath: Path) -> List[BenchmarkResult]:
    """Load benchmark results from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    return [BenchmarkResult(**r) for r in data]
