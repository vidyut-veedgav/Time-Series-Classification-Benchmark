"""
Data loading utilities for time series experiments.
Supports classification, forecasting, and anomaly detection tasks.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).parent.parent / "data"


class TimeSeriesClassificationDataset(Dataset):
    """Dataset for time series classification tasks."""

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


class TimeSeriesForecastingDataset(Dataset):
    """Dataset for time series forecasting tasks."""

    def __init__(
        self,
        data: np.ndarray,
        seq_length: int = 96,
        pred_length: int = 24,
        stride: int = 1
    ):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.stride = stride

        self.total_length = seq_length + pred_length
        self.n_samples = (len(data) - self.total_length) // stride + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self.data[start:start + self.seq_length]
        y = self.data[start + self.seq_length:start + self.total_length]

        return torch.FloatTensor(x), torch.FloatTensor(y)


class TimeSeriesAnomalyDataset(Dataset):
    """Dataset for time series anomaly detection."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: int = 64,
        stride: int = 1
    ):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride

        self.n_samples = (len(data) - window_size) // stride + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self.data[start:start + self.window_size]
        y = self.labels[start:start + self.window_size]

        return torch.FloatTensor(x), torch.FloatTensor(y)


def load_ucr_dataset(
    dataset_name: str = "ECG200",
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Load a UCR time series classification dataset.

    Returns dict with train_data, train_labels, test_data, test_labels, n_classes, seq_length
    """
    dataset_dir = DATA_DIR / "ucr" / dataset_name

    train_file = dataset_dir / f"{dataset_name}_TRAIN.tsv"
    test_file = dataset_dir / f"{dataset_name}_TEST.tsv"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_name} not found. Run download_data.py first."
        )

    train_df = pd.read_csv(train_file, sep="\t", header=None)
    test_df = pd.read_csv(test_file, sep="\t", header=None)

    train_labels = train_df.iloc[:, 0].values
    train_data = train_df.iloc[:, 1:].values

    test_labels = test_df.iloc[:, 0].values
    test_data = test_df.iloc[:, 1:].values

    unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    train_labels = np.array([label_map[l] for l in train_labels])
    test_labels = np.array([label_map[l] for l in test_labels])

    if normalize:
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data.T).T
        test_data = scaler.transform(test_data.T).T

    train_data = train_data[:, :, np.newaxis]
    test_data = test_data[:, :, np.newaxis]

    return {
        "train_data": train_data,
        "train_labels": train_labels,
        "test_data": test_data,
        "test_labels": test_labels,
        "n_classes": len(unique_labels),
        "seq_length": train_data.shape[1],
        "n_features": 1
    }


def load_ett_dataset(
    filename: str = "ETTh1.csv",
    target_col: str = "OT",
    normalize: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1
) -> Dict[str, Any]:
    """
    Load ETT dataset for forecasting.

    Returns dict with train/val/test data, scaler, and metadata
    """
    filepath = DATA_DIR / "ett" / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"ETT dataset {filename} not found. Run download_data.py first."
        )

    df = pd.read_csv(filepath)

    feature_cols = [c for c in df.columns if c not in ["date", target_col]]
    feature_cols.append(target_col)

    data = df[feature_cols].values

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    scaler = None
    if normalize:
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)

    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "scaler": scaler,
        "n_features": len(feature_cols),
        "feature_names": feature_cols
    }


def load_anomaly_dataset(
    filename: str = "synthetic_anomaly.csv",
    normalize: bool = True,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
) -> Dict[str, Any]:
    """
    Load anomaly detection dataset.

    Returns dict with train/val/test data and labels
    """
    filepath = DATA_DIR / "anomaly" / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Anomaly dataset {filename} not found. Run download_data.py first."
        )

    df = pd.read_csv(filepath)

    values = df["value"].values.reshape(-1, 1)
    labels = df["label"].values

    n = len(values)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    scaler = None
    if normalize:
        scaler = StandardScaler()
        values = scaler.fit_transform(values)

    return {
        "train_data": values[:train_end],
        "train_labels": labels[:train_end],
        "val_data": values[train_end:val_end],
        "val_labels": labels[train_end:val_end],
        "test_data": values[val_end:],
        "test_labels": labels[val_end:],
        "scaler": scaler
    }


def create_classification_loaders(
    dataset_name: str = "ECG200",
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Create train and test data loaders for classification."""
    data = load_ucr_dataset(dataset_name)

    train_dataset = TimeSeriesClassificationDataset(
        data["train_data"], data["train_labels"]
    )
    test_dataset = TimeSeriesClassificationDataset(
        data["test_data"], data["test_labels"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, data


def create_forecasting_loaders(
    filename: str = "ETTh1.csv",
    seq_length: int = 96,
    pred_length: int = 24,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Create train, val, and test data loaders for forecasting."""
    data = load_ett_dataset(filename)

    train_dataset = TimeSeriesForecastingDataset(
        data["train_data"], seq_length, pred_length
    )
    val_dataset = TimeSeriesForecastingDataset(
        data["val_data"], seq_length, pred_length
    )
    test_dataset = TimeSeriesForecastingDataset(
        data["test_data"], seq_length, pred_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, data


def create_anomaly_loaders(
    window_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Create data loaders for anomaly detection."""
    data = load_anomaly_dataset()

    train_dataset = TimeSeriesAnomalyDataset(
        data["train_data"], data["train_labels"], window_size
    )
    val_dataset = TimeSeriesAnomalyDataset(
        data["val_data"], data["val_labels"], window_size
    )
    test_dataset = TimeSeriesAnomalyDataset(
        data["test_data"], data["test_labels"], window_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, data


def load_medical_dataset(normalize: bool = True) -> Dict[str, Any]:
    """
    Load MIT-BIH Arrhythmia Database for classification.

    Classes: Normal, Supraventricular, Ventricular
    Returns dict with train_data, train_labels, test_data, test_labels, n_classes, seq_length
    """
    dataset_dir = DATA_DIR / "medical" / "mitbih"

    train_file = dataset_dir / "mitbih_train.csv"
    test_file = dataset_dir / "mitbih_test.csv"

    if not train_file.exists():
        raise FileNotFoundError(
            "MIT-BIH dataset not found. Run: python -c \"from src.download_data import download_mitbih_dataset; download_mitbih_dataset()\""
        )

    train_df = pd.read_csv(train_file, header=None)
    test_df = pd.read_csv(test_file, header=None)

    train_labels = train_df.iloc[:, 0].values.astype(int)
    train_data = train_df.iloc[:, 1:].values

    test_labels = test_df.iloc[:, 0].values.astype(int)
    test_data = test_df.iloc[:, 1:].values

    if normalize:
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data.T).T
        test_data = scaler.transform(test_data.T).T

    train_data = train_data[:, :, np.newaxis]
    test_data = test_data[:, :, np.newaxis]

    n_classes = len(np.unique(np.concatenate([train_labels, test_labels])))

    return {
        "train_data": train_data,
        "train_labels": train_labels,
        "test_data": test_data,
        "test_labels": test_labels,
        "n_classes": n_classes,
        "seq_length": train_data.shape[1],
        "n_features": 1
    }


def create_medical_loaders(
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Create train and test data loaders for medical ECG classification."""
    data = load_medical_dataset()

    train_dataset = TimeSeriesClassificationDataset(
        data["train_data"], data["train_labels"]
    )
    test_dataset = TimeSeriesClassificationDataset(
        data["test_data"], data["test_labels"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, data
