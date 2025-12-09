"""Model loading and inference utilities for visual applications."""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

from src.models import LSTMClassifier, RNNClassifier, CNNClassifier, VAEClassifier


@dataclass
class DemoModel:
    """Container for loaded model and metadata."""
    name: str
    model: torch.nn.Module
    hyperparams: Dict[str, Any]
    metrics: Dict[str, float]
    device: str


def load_demo_model(
    model_name: str,
    checkpoint_path: Path,
    device: str = "auto"
) -> Optional[DemoModel]:
    """
    Load a single pre-trained model from checkpoint.

    Args:
        model_name: Name of model ("LSTM", "RNN", "CNN", "VAE")
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on ("auto", "cuda", "cpu")

    Returns:
        DemoModel instance or None if checkpoint doesn't exist
    """
    if not checkpoint_path.exists():
        return None

    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract info
    hyperparams = checkpoint["hyperparams"]
    metrics = checkpoint["metrics"]
    data_info = checkpoint["data_info"]

    # Instantiate model based on name
    model_classes = {
        "LSTM": LSTMClassifier,
        "RNN": RNNClassifier,
        "CNN": CNNClassifier,
        "VAE": VAEClassifier
    }

    model_class = model_classes[model_name]

    # Create model instance
    model = model_class(
        n_features=data_info["n_features"],
        seq_length=data_info["seq_length"],
        n_classes=data_info["n_classes"],
        **hyperparams
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return DemoModel(
        name=model_name,
        model=model,
        hyperparams=hyperparams,
        metrics=metrics,
        device=device
    )


def load_all_demo_models(
    checkpoint_dir: Path = Path("checkpoints/ecg200_best"),
    device: str = "auto"
) -> Dict[str, DemoModel]:
    """
    Load all available pre-trained models.

    Args:
        checkpoint_dir: Directory containing model checkpoints
        device: Device to load models on

    Returns:
        Dictionary mapping model name to DemoModel instance
    """
    models = {}

    for model_name in ["LSTM", "RNN", "CNN", "VAE"]:
        checkpoint_path = checkpoint_dir / f"{model_name.lower()}.pt"
        model = load_demo_model(model_name, checkpoint_path, device)
        if model is not None:
            models[model_name] = model

    return models


def predict_single_sample(
    demo_model: DemoModel,
    sample: np.ndarray,
    return_probabilities: bool = True
) -> Dict[str, Any]:
    """
    Run inference on a single time series sample.

    Args:
        demo_model: DemoModel instance
        sample: Time series sample, shape (seq_length, n_features) or (seq_length,)
        return_probabilities: Whether to return class probabilities

    Returns:
        Dictionary with prediction results:
            - predicted_class: int
            - probabilities: np.ndarray (if return_probabilities=True)
            - confidence: float (max probability)
            - logits: np.ndarray
    """
    model = demo_model.model
    device = demo_model.device

    # Ensure sample has correct shape
    if sample.ndim == 1:
        sample = sample.reshape(-1, 1)  # (seq_length,) -> (seq_length, 1)

    # Add batch dimension: (seq_length, n_features) -> (1, seq_length, n_features)
    sample_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        logits = model(sample_tensor)  # Shape: (1, n_classes)

    # Convert to numpy
    logits_np = logits.cpu().numpy()[0]  # Shape: (n_classes,)

    # Get predicted class
    predicted_class = int(logits_np.argmax())

    # Calculate probabilities using softmax
    if return_probabilities:
        exp_logits = np.exp(logits_np - np.max(logits_np))  # Numerical stability
        probabilities = exp_logits / exp_logits.sum()
        confidence = float(probabilities[predicted_class])
    else:
        probabilities = None
        confidence = None

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "confidence": confidence,
        "logits": logits_np
    }


def predict_batch(
    demo_model: DemoModel,
    samples: np.ndarray,
    batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Run inference on multiple samples efficiently.

    Args:
        demo_model: DemoModel instance
        samples: Time series samples, shape (n_samples, seq_length, n_features)
        batch_size: Batch size for inference

    Returns:
        Dictionary with:
            - predictions: np.ndarray (n_samples,)
            - probabilities: np.ndarray (n_samples, n_classes)
            - confidences: np.ndarray (n_samples,)
    """
    model = demo_model.model
    device = demo_model.device

    n_samples = len(samples)
    all_predictions = []
    all_probabilities = []

    # Process in batches
    for i in range(0, n_samples, batch_size):
        batch = samples[i:i+batch_size]
        batch_tensor = torch.FloatTensor(batch).to(device)

        with torch.no_grad():
            logits = model(batch_tensor)  # Shape: (batch_size, n_classes)

        # Convert to probabilities
        logits_np = logits.cpu().numpy()
        exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        preds = logits_np.argmax(axis=1)

        all_predictions.append(preds)
        all_probabilities.append(probs)

    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    confidences = probabilities.max(axis=1)

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "confidences": confidences
    }
