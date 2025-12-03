"""
Download datasets for time series experiments.
Includes UCR archive, ETT, and synthetic data generation.
"""

import os
import requests
import zipfile
import numpy as np
import pandas as pd
from io import BytesIO
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def download_ucr_dataset(dataset_name: str = "ECG200") -> Path:
    """
    Download a dataset from the UCR Time Series Archive.
    Popular datasets: ECG200, ECG5000, FordA, Wafer, GunPoint, Coffee
    """
    base_url = "https://www.timeseriesclassification.com/aeon-toolkit"
    url = f"{base_url}/{dataset_name}.zip"

    dataset_dir = DATA_DIR / "ucr" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_file = dataset_dir / f"{dataset_name}_TRAIN.tsv"
    test_file = dataset_dir / f"{dataset_name}_TEST.tsv"

    if train_file.exists() and test_file.exists():
        print(f"Dataset {dataset_name} already exists")
        return dataset_dir

    print(f"Downloading {dataset_name} from UCR archive")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            for member in zf.namelist():
                if member.endswith(".tsv") or member.endswith(".ts"):
                    filename = os.path.basename(member)
                    with open(dataset_dir / filename, "wb") as f:
                        f.write(zf.read(member))

        print(f"Successfully downloaded {dataset_name}")
        return dataset_dir

    except Exception as e:
        print(f"Failed to download {dataset_name}: {e}")
        print("Generating synthetic classification data instead")
        return generate_synthetic_classification(dataset_name, dataset_dir)


def generate_synthetic_classification(name: str, save_dir: Path) -> Path:
    """Generate synthetic time series classification data."""
    save_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_samples_train = 100
    n_samples_test = 50
    seq_length = 96
    n_classes = 2

    def generate_class_data(n_samples, class_id):
        data = []
        for _ in range(n_samples):
            t = np.linspace(0, 4 * np.pi, seq_length)
            if class_id == 0:
                signal = np.sin(t) + 0.1 * np.random.randn(seq_length)
            else:
                signal = np.sin(2 * t) + 0.1 * np.random.randn(seq_length)
            data.append(signal)
        return np.array(data)

    train_data = []
    train_labels = []
    for c in range(n_classes):
        class_data = generate_class_data(n_samples_train // n_classes, c)
        train_data.append(class_data)
        train_labels.extend([c] * len(class_data))

    train_data = np.vstack(train_data)
    train_labels = np.array(train_labels)

    test_data = []
    test_labels = []
    for c in range(n_classes):
        class_data = generate_class_data(n_samples_test // n_classes, c)
        test_data.append(class_data)
        test_labels.extend([c] * len(class_data))

    test_data = np.vstack(test_data)
    test_labels = np.array(test_labels)

    train_df = pd.DataFrame(
        np.column_stack([train_labels, train_data])
    )
    test_df = pd.DataFrame(
        np.column_stack([test_labels, test_data])
    )

    train_df.to_csv(save_dir / f"{name}_TRAIN.tsv", sep="\t", index=False, header=False)
    test_df.to_csv(save_dir / f"{name}_TEST.tsv", sep="\t", index=False, header=False)

    print(f"Generated synthetic classification data at {save_dir}")
    return save_dir


def download_ett_dataset() -> Path:
    """
    Download ETT (Electricity Transformer Temperature) dataset.
    Used for time series forecasting benchmarks.
    """
    dataset_dir = DATA_DIR / "ett"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    files = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]
    base_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"

    for filename in files:
        filepath = dataset_dir / filename
        if filepath.exists():
            print(f"{filename} already exists")
            continue

        url = f"{base_url}/{filename}"
        print(f"Downloading {filename}")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Successfully downloaded {filename}")

        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            print("Generating synthetic forecasting data instead")
            generate_synthetic_forecasting(filepath)

    return dataset_dir


def generate_synthetic_forecasting(filepath: Path) -> None:
    """Generate synthetic time series forecasting data."""
    np.random.seed(42)

    n_points = 17420
    t = np.arange(n_points)

    dates = pd.date_range(start="2016-07-01", periods=n_points, freq="h")

    trend = 0.001 * t
    daily = 5 * np.sin(2 * np.pi * t / 24)
    weekly = 3 * np.sin(2 * np.pi * t / (24 * 7))
    yearly = 10 * np.sin(2 * np.pi * t / (24 * 365))
    noise = np.random.randn(n_points)

    oil_temp = 50 + trend + daily + weekly + yearly + noise

    data = {
        "date": dates,
        "HUFL": oil_temp + np.random.randn(n_points) * 2,
        "HULL": oil_temp - 10 + np.random.randn(n_points) * 2,
        "MUFL": oil_temp + 5 + np.random.randn(n_points) * 1.5,
        "MULL": oil_temp - 5 + np.random.randn(n_points) * 1.5,
        "LUFL": oil_temp + 3 + np.random.randn(n_points) * 1,
        "LULL": oil_temp - 3 + np.random.randn(n_points) * 1,
        "OT": oil_temp,
    }

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Generated synthetic forecasting data at {filepath}")


def download_mitbih_dataset() -> Path:
    """
    Download MIT-BIH Arrhythmia Database from PhysioNet.

    Contains 48 half-hour excerpts of two-channel ambulatory ECG recordings
    with beat-by-beat annotations. This is one of the most widely used
    datasets for arrhythmia classification research.

    Source: https://physionet.org/content/mitdb/
    """
    import wfdb
    from tqdm import tqdm

    dataset_dir = DATA_DIR / "medical" / "mitbih"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_file = dataset_dir / "mitbih_train.csv"
    test_file = dataset_dir / "mitbih_test.csv"

    if train_file.exists() and test_file.exists():
        print("MIT-BIH dataset already exists")
        return dataset_dir

    print("Downloading MIT-BIH Arrhythmia Database from PhysioNet")

    record_ids = [
        "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
        "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
        "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
        "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
        "222", "223", "228", "230", "231", "232", "233", "234"
    ]

    beat_labels = {
        "N": 0,  # Normal beat
        "L": 0,  # Left bundle branch block beat (treat as normal variant)
        "R": 0,  # Right bundle branch block beat (treat as normal variant)
        "A": 1,  # Atrial premature beat
        "a": 1,  # Aberrated atrial premature beat
        "S": 1,  # Supraventricular premature beat
        "V": 2,  # Premature ventricular contraction
        "F": 2,  # Fusion of ventricular and normal beat
    }

    label_names = {0: "Normal", 1: "Supraventricular", 2: "Ventricular"}

    window_size = 360
    half_window = window_size // 2

    all_beats = []
    all_labels = []

    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    print("Downloading and processing records")
    for record_id in tqdm(record_ids, desc="Processing records"):
        try:
            record = wfdb.rdrecord(record_id, pn_dir="mitdb", sampto=650000)
            annotation = wfdb.rdann(record_id, "atr", pn_dir="mitdb", sampto=650000)

            signal = record.p_signal[:, 0]

            for idx, symbol in zip(annotation.sample, annotation.symbol):
                if symbol not in beat_labels:
                    continue

                start = idx - half_window
                end = idx + half_window

                if start < 0 or end > len(signal):
                    continue

                beat = signal[start:end]

                beat = (beat - beat.mean()) / (beat.std() + 1e-8)

                all_beats.append(beat)
                all_labels.append(beat_labels[symbol])

        except Exception as e:
            print(f"  Skipping record {record_id}: {e}")
            continue

    all_beats = np.array(all_beats)
    all_labels = np.array(all_labels)

    print(f"Extracted {len(all_beats)} beats")
    for label_id, label_name in label_names.items():
        count = np.sum(all_labels == label_id)
        print(f"  {label_name}: {count}")

    np.random.seed(42)
    indices = np.random.permutation(len(all_beats))
    all_beats = all_beats[indices]
    all_labels = all_labels[indices]

    split_idx = int(len(all_beats) * 0.8)
    train_data = all_beats[:split_idx]
    train_labels = all_labels[:split_idx]
    test_data = all_beats[split_idx:]
    test_labels = all_labels[split_idx:]

    train_df = pd.DataFrame(
        np.column_stack([train_labels, train_data])
    )
    test_df = pd.DataFrame(
        np.column_stack([test_labels, test_data])
    )

    train_df.to_csv(train_file, index=False, header=False)
    test_df.to_csv(test_file, index=False, header=False)

    with open(dataset_dir / "label_map.txt", "w") as f:
        for label_id, label_name in label_names.items():
            f.write(f"{label_id}: {label_name}\n")

    print(f"Saved MIT-BIH dataset to {dataset_dir}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")

    return dataset_dir


def generate_synthetic_anomaly_data() -> Path:
    """Generate synthetic data for anomaly detection."""
    dataset_dir = DATA_DIR / "anomaly"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_points = 5000
    t = np.arange(n_points)

    signal = np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 25)
    signal += 0.1 * np.random.randn(n_points)

    labels = np.zeros(n_points)
    anomaly_indices = [500, 1200, 2000, 2800, 3500, 4200]
    for idx in anomaly_indices:
        signal[idx:idx+20] += np.random.randn(20) * 3
        labels[idx:idx+20] = 1

    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=n_points, freq="min"),
        "value": signal,
        "label": labels.astype(int)
    })

    filepath = dataset_dir / "synthetic_anomaly.csv"
    df.to_csv(filepath, index=False)
    print(f"Generated synthetic anomaly data at {filepath}")

    return dataset_dir


def download_all_datasets():
    """Download all datasets for experiments."""
    print("Downloading UCR datasets")
    ucr_datasets = ["ECG200", "ECG5000", "FordA"]
    for name in ucr_datasets:
        download_ucr_dataset(name)

    print("\nDownloading ETT dataset")
    download_ett_dataset()

    print("\nDownloading MIT-BIH Arrhythmia Database")
    download_mitbih_dataset()

    print("\nGenerating anomaly detection data")
    generate_synthetic_anomaly_data()

    print("\nAll datasets ready")


if __name__ == "__main__":
    download_all_datasets()
