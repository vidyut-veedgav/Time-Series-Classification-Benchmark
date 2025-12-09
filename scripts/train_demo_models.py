#!/usr/bin/env python3
"""
Train and save best models for visual demo.

Usage:
    python scripts/train_demo_models.py

    Optional arguments:
    --models LSTM RNN CNN VAE  (default: LSTM for MVP)
    --dataset ECG200  (default: ECG200)
    --epochs 50  (default: 50)
"""

import argparse
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import create_classification_loaders
from src.models import LSTMClassifier, RNNClassifier, CNNClassifier, VAEClassifier
from src.benchmark import benchmark_classification


# Best configurations from results/best_configurations.csv
BEST_CONFIGS = {
    "LSTM": {
        "model_class": LSTMClassifier,
        "hyperparams": {
            "hidden_size": 64,
            "n_layers": 2,
            "bidirectional": False,
            "dropout": 0.1
        }
    },
    "RNN": {
        "model_class": RNNClassifier,
        "hyperparams": {
            "hidden_size": 64,
            "n_layers": 2,
            "use_gru": True,
            "bidirectional": True,
            "dropout": 0.1
        }
    },
    "CNN": {
        "model_class": CNNClassifier,
        "hyperparams": {
            "n_filters": 64,
            "n_blocks": 2,
            "kernel_size": 3,
            "dropout": 0.1
        }
    },
    "VAE": {
        "model_class": VAEClassifier,
        "hyperparams": {
            "hidden_size": 64,
            "latent_dim": 16,
            "n_layers": 1,
            "dropout": 0.1
        }
    }
}


def train_and_save_demo_models(
    dataset_name: str = "ECG200",
    model_names: list = None,
    checkpoint_dir: Path = Path("checkpoints/ecg200_best"),
    n_epochs: int = 50,
    device: str = "auto"
):
    """Train best configuration for each model and save checkpoints."""

    if model_names is None:
        model_names = ["LSTM"]  # MVP: Just LSTM

    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Training models: {', '.join(model_names)}")
    print(f"Dataset: {dataset_name}")
    print(f"Epochs: {n_epochs}")
    print("-" * 60)

    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading {dataset_name} dataset...")
    train_loader, test_loader, data_info = create_classification_loaders(
        dataset_name=dataset_name,
        batch_size=32
    )

    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Sequence length: {data_info['seq_length']}")
    print(f"  Features: {data_info['n_features']}")
    print(f"  Classes: {data_info['n_classes']}")

    # Train each model
    for model_name in model_names:
        if model_name not in BEST_CONFIGS:
            print(f"\nWarning: Unknown model '{model_name}', skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")

        config = BEST_CONFIGS[model_name]

        # Instantiate model
        model = config["model_class"](
            n_features=data_info["n_features"],
            seq_length=data_info["seq_length"],
            n_classes=data_info["n_classes"],
            **config["hyperparams"]
        )

        print(f"Model architecture: {model.__class__.__name__}")
        print(f"Hyperparameters: {config['hyperparams']}")

        # Train
        result = benchmark_classification(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=model_name,
            dataset_name=dataset_name,
            n_epochs=n_epochs,
            device=device
        )

        # Print results
        print(f"\nTraining completed!")
        print(f"  Accuracy: {result.metrics['accuracy']:.2%}")
        print(f"  F1 Score: {result.metrics['f1_macro']:.4f}")
        print(f"  Precision: {result.metrics['precision']:.4f}")
        print(f"  Recall: {result.metrics['recall']:.4f}")
        print(f"  Train time: {result.train_time:.2f}s")
        print(f"  Inference time: {result.inference_time:.4f}s")
        print(f"  Parameters: {result.n_parameters:,}")

        # Save checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "hyperparams": config["hyperparams"],
            "metrics": result.metrics,
            "data_info": data_info,
            "train_time": result.train_time,
            "inference_time": result.inference_time,
            "n_parameters": result.n_parameters
        }

        checkpoint_path = checkpoint_dir / f"{model_name.lower()}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved to: {checkpoint_path}")

    print(f"\n{'='*60}")
    print("All models trained and saved successfully!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train demo models for visual applications")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["LSTM"],
        choices=["LSTM", "RNN", "CNN", "VAE"],
        help="Models to train (default: LSTM for MVP)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ECG200",
        help="Dataset name (default: ECG200)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/ecg200_best"),
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)"
    )

    args = parser.parse_args()

    train_and_save_demo_models(
        dataset_name=args.dataset,
        model_names=args.models,
        checkpoint_dir=args.checkpoint_dir,
        n_epochs=args.epochs,
        device=args.device
    )


if __name__ == "__main__":
    main()
