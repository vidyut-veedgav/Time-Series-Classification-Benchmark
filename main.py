"""
Main script to run all time series experiments.

Usage:
    python main.py --task classification --dataset ECG200
    python main.py --task forecasting --dataset ETTh1
    python main.py --task anomaly
    python main.py --task medical
    python main.py --task all
"""

import argparse
import torch
from pathlib import Path

from src.download_data import download_all_datasets
from src.data_loader import (
    create_classification_loaders,
    create_forecasting_loaders,
    create_anomaly_loaders,
    create_medical_loaders
)
from src.models import (
    TransformerClassifier,
    TransformerForecaster,
    LSTMClassifier,
    LSTMForecaster,
    CNNClassifier,
    CNNForecaster,
    RNNClassifier,
    RNNForecaster,
    VAEAnomalyDetector,
    STLClassifier,
    STLForecaster,
    ImageCNNClassifier,
    MultiScaleImageCNN
)
from src.benchmark import (
    benchmark_classification,
    benchmark_forecasting,
    benchmark_anomaly_detection,
    save_results
)
from src.plots import (
    plot_training_curves,
    plot_metric_comparison,
    plot_all_metrics,
    plot_time_comparison,
    create_summary_table
)


RESULTS_DIR = Path(__file__).parent / "results"


def run_classification_experiments(dataset_name: str = "ECG200", n_epochs: int = 50):
    """Run classification experiments on all models."""
    print(f"Running classification experiments on {dataset_name}")

    train_loader, test_loader, data_info = create_classification_loaders(
        dataset_name, batch_size=32
    )

    n_features = data_info["n_features"]
    seq_length = data_info["seq_length"]
    n_classes = data_info["n_classes"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    models = {
        "Transformer": TransformerClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            d_model=64,
            n_heads=4,
            n_layers=2
        ),
        "LSTM": LSTMClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            hidden_size=64,
            n_layers=2
        ),
        "CNN": CNNClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            n_filters=64,
            n_blocks=3
        ),
        "RNN": RNNClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            hidden_size=64,
            n_layers=2
        ),
        "STL": STLClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            hidden_size=64,
            period=min(24, seq_length // 4)
        ),
        "ImageCNN": ImageCNNClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            image_size=64,
            transform_method="spectrogram"
        )
    }

    results = []
    for name, model in models.items():
        print(f"\nBenchmarking {name}")
        result = benchmark_classification(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=name,
            dataset_name=dataset_name,
            n_epochs=n_epochs,
            device=device
        )
        results.append(result)
        print(f"{name} accuracy: {result.metrics['accuracy']:.4f}")

    save_results(results, RESULTS_DIR / f"classification_{dataset_name}.json")

    print("\nResults summary:")
    print(create_summary_table(results))

    plot_training_curves(results, RESULTS_DIR / f"training_curves_{dataset_name}.png")
    plot_metric_comparison(results, "accuracy", RESULTS_DIR / f"accuracy_{dataset_name}.png")
    plot_time_comparison(results, RESULTS_DIR / f"time_comparison_{dataset_name}.png")

    return results


def run_forecasting_experiments(dataset_name: str = "ETTh1.csv", n_epochs: int = 50):
    """Run forecasting experiments on all models."""
    print(f"Running forecasting experiments on {dataset_name}")

    seq_length = 96
    pred_length = 24

    train_loader, val_loader, test_loader, data_info = create_forecasting_loaders(
        filename=dataset_name,
        seq_length=seq_length,
        pred_length=pred_length,
        batch_size=32
    )

    n_features = data_info["n_features"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    models = {
        "Transformer": TransformerForecaster(
            n_features=n_features,
            seq_length=seq_length,
            pred_length=pred_length,
            d_model=64,
            n_heads=4
        ),
        "LSTM": LSTMForecaster(
            n_features=n_features,
            seq_length=seq_length,
            pred_length=pred_length,
            hidden_size=64,
            n_layers=2
        ),
        "CNN": CNNForecaster(
            n_features=n_features,
            seq_length=seq_length,
            pred_length=pred_length,
            n_filters=64,
            n_layers=4
        ),
        "RNN": RNNForecaster(
            n_features=n_features,
            seq_length=seq_length,
            pred_length=pred_length,
            hidden_size=64,
            n_layers=2
        ),
        "STL": STLForecaster(
            n_features=n_features,
            seq_length=seq_length,
            pred_length=pred_length,
            hidden_size=64,
            period=24
        )
    }

    results = []
    for name, model in models.items():
        print(f"\nBenchmarking {name}")
        result = benchmark_forecasting(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_name=name,
            dataset_name=dataset_name,
            n_epochs=n_epochs,
            device=device
        )
        results.append(result)
        print(f"{name} MSE: {result.metrics['mse']:.4f}")

    dataset_base = dataset_name.replace(".csv", "")
    save_results(results, RESULTS_DIR / f"forecasting_{dataset_base}.json")

    print("\nResults summary:")
    print(create_summary_table(results))

    plot_training_curves(results, RESULTS_DIR / f"training_curves_{dataset_base}.png")
    plot_metric_comparison(results, "mse", RESULTS_DIR / f"mse_{dataset_base}.png")
    plot_metric_comparison(results, "mae", RESULTS_DIR / f"mae_{dataset_base}.png")

    return results


def run_anomaly_experiments(n_epochs: int = 50):
    """Run anomaly detection experiments."""
    print("Running anomaly detection experiments")

    window_size = 64

    train_loader, val_loader, test_loader, data_info = create_anomaly_loaders(
        window_size=window_size,
        batch_size=32
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = VAEAnomalyDetector(
        n_features=1,
        seq_length=window_size,
        hidden_size=64,
        latent_dim=16,
        n_layers=2
    )

    print("\nBenchmarking VAE")
    result = benchmark_anomaly_detection(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="VAE",
        dataset_name="synthetic_anomaly",
        n_epochs=n_epochs,
        device=device
    )

    print(f"VAE F1: {result.metrics['f1']:.4f}")

    save_results([result], RESULTS_DIR / "anomaly_detection.json")

    print("\nResults summary:")
    print(create_summary_table([result]))

    return [result]


def run_medical_experiments(n_epochs: int = 50):
    """Run medical ECG classification experiments."""
    print("Running medical ECG classification experiments")

    train_loader, test_loader, data_info = create_medical_loaders(batch_size=32)

    n_features = data_info["n_features"]
    seq_length = data_info["seq_length"]
    n_classes = data_info["n_classes"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Classes: {n_classes}, Sequence length: {seq_length}")

    models = {
        "Transformer": TransformerClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            d_model=64,
            n_heads=4,
            n_layers=2
        ),
        "LSTM": LSTMClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            hidden_size=64,
            n_layers=2
        ),
        "CNN": CNNClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            n_filters=64,
            n_blocks=3
        ),
        "RNN": RNNClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            hidden_size=64,
            n_layers=2
        ),
        "ImageCNN_Spectrogram": ImageCNNClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            image_size=64,
            transform_method="spectrogram"
        ),
        "ImageCNN_GAF": ImageCNNClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            image_size=64,
            transform_method="gaf"
        ),
        "MultiScaleImageCNN": MultiScaleImageCNN(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            image_size=64
        )
    }

    results = []
    for name, model in models.items():
        print(f"\nBenchmarking {name}")
        result = benchmark_classification(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=name,
            dataset_name="medical_ecg",
            n_epochs=n_epochs,
            device=device
        )
        results.append(result)
        print(f"{name} accuracy: {result.metrics['accuracy']:.4f}")

    save_results(results, RESULTS_DIR / "medical_ecg.json")

    print("\nResults summary:")
    print(create_summary_table(results))

    plot_training_curves(results, RESULTS_DIR / "training_curves_medical.png")
    plot_metric_comparison(results, "accuracy", RESULTS_DIR / "accuracy_medical.png")
    plot_all_metrics(results, RESULTS_DIR / "all_metrics_medical.png")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run time series experiments")
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "forecasting", "anomaly", "medical", "all"],
        default="all",
        help="Task type to run"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g., ECG200 for classification, ETTh1.csv for forecasting)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download datasets before running experiments"
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.download:
        print("Downloading datasets")
        download_all_datasets()

    if args.task == "classification" or args.task == "all":
        dataset = args.dataset if args.dataset and args.task == "classification" else "ECG200"
        run_classification_experiments(dataset, args.epochs)

    if args.task == "forecasting" or args.task == "all":
        dataset = args.dataset if args.dataset and args.task == "forecasting" else "ETTh1.csv"
        run_forecasting_experiments(dataset, args.epochs)

    if args.task == "anomaly" or args.task == "all":
        run_anomaly_experiments(args.epochs)

    if args.task == "medical" or args.task == "all":
        run_medical_experiments(args.epochs)

    print("\nAll experiments completed")


if __name__ == "__main__":
    main()
