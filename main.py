"""
Comprehensive Time Series Classification Benchmark Script.

Orchestrates experiments across 4 models (LSTM, RNN, CNN, VAE) on 2 datasets (FordA, ECG200)
with multiple hyperparameter configurations. Generates detailed results tables and visualizations
including cross-dataset comparisons.

Usage:
    python main3.py --epochs 50
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product
from typing import List, Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.data_loader import create_classification_loaders
from src.models.lstm import LSTMClassifier
from src.models.rnn import RNNClassifier
from src.models.cnn import CNNClassifier
from src.models.vae import VAEClassifier
from src.benchmark import benchmark_classification, count_parameters, BenchmarkResult


# Setup directories
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def setup_directories():
    """Create results and plots directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Hyperparameter Configurations
# Reduced to 4 combinations per model × 2 datasets = 8 trials per experiment
# ============================================================================

LSTM_HYPERPARAMS = {
    'hidden_size': [64, 128],
    'n_layers': [1, 2],
    'bidirectional': [False]
}
# Combinations: 2 × 2 × 1 = 4 trials per dataset, 8 total

RNN_HYPERPARAMS = {
    'hidden_size': [64, 128],
    'n_layers': [1, 2],
    'use_gru': [True]
}
# Combinations: 2 × 2 × 1 = 4 trials per dataset, 8 total

CNN_HYPERPARAMS = {
    'n_filters': [32, 64],
    'n_blocks': [2, 3],
    'kernel_size': [3]
}
# Combinations: 2 × 2 × 1 = 4 trials per dataset, 8 total

VAE_HYPERPARAMS = {
    'hidden_size': [32, 64],
    'latent_dim': [8, 16],
    'n_layers': [1]
}
# Combinations: 2 × 2 × 1 = 4 trials per dataset, 8 total


# ============================================================================
# Experiment Functions
# ============================================================================

def run_lstm_experiment(dataset_name: str, n_epochs: int = 50) -> List[Dict[str, Any]]:
    """
    Run LSTM classification experiment with hyperparameter variations.

    Args:
        dataset_name: Name of dataset (FordA or ECG200)
        n_epochs: Number of training epochs

    Returns:
        List of result dictionaries with metrics and hyperparameters
    """
    print(f"\n{'='*80}")
    print(f"Running LSTM Experiment on {dataset_name}")
    print(f"{'='*80}")

    # Load dataset
    train_loader, test_loader, data_info = create_classification_loaders(
        dataset_name, batch_size=32
    )

    n_features = data_info["n_features"]
    seq_length = data_info["seq_length"]
    n_classes = data_info["n_classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Dataset info: features={n_features}, seq_length={seq_length}, classes={n_classes}")
    print(f"Device: {device}")

    # Generate hyperparameter combinations
    hyperparam_keys = list(LSTM_HYPERPARAMS.keys())
    hyperparam_values = list(LSTM_HYPERPARAMS.values())
    combinations = list(product(*hyperparam_values))

    results = []

    for idx, combo in enumerate(combinations, 1):
        hyperparams = dict(zip(hyperparam_keys, combo))
        print(f"\n[{idx}/{len(combinations)}] Testing LSTM with: {hyperparams}")

        # Create model
        model = LSTMClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            hidden_size=hyperparams['hidden_size'],
            n_layers=hyperparams['n_layers'],
            bidirectional=hyperparams['bidirectional'],
            dropout=0.1
        )

        # Benchmark
        result = benchmark_classification(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=f"LSTM_{hyperparams}",
            dataset_name=dataset_name,
            n_epochs=n_epochs,
            device=device
        )

        # Store results with hyperparameters
        result_dict = {
            'model': 'LSTM',
            'dataset': dataset_name,
            'hyperparams': str(hyperparams),
            'hidden_size': hyperparams['hidden_size'],
            'n_layers': hyperparams['n_layers'],
            'bidirectional': hyperparams['bidirectional'],
            'accuracy': result.metrics['accuracy'],
            'precision': result.metrics['precision'],
            'recall': result.metrics['recall'],
            'f1_macro': result.metrics['f1_macro'],
            'f1_weighted': result.metrics['f1_weighted'],
            'train_time': result.train_time,
            'inference_time': result.inference_time,
            'n_parameters': result.n_parameters,
            'threshold': None,
            'epoch_history': result.epoch_history
        }

        results.append(result_dict)
        print(f"Accuracy: {result.metrics['accuracy']:.4f}, F1: {result.metrics['f1_macro']:.4f}")

    return results


def run_rnn_experiment(dataset_name: str, n_epochs: int = 50) -> List[Dict[str, Any]]:
    """
    Run RNN classification experiment with hyperparameter variations.

    Args:
        dataset_name: Name of dataset (FordA or ECG200)
        n_epochs: Number of training epochs

    Returns:
        List of result dictionaries with metrics and hyperparameters
    """
    print(f"\n{'='*80}")
    print(f"Running RNN Experiment on {dataset_name}")
    print(f"{'='*80}")

    # Load dataset
    train_loader, test_loader, data_info = create_classification_loaders(
        dataset_name, batch_size=32
    )

    n_features = data_info["n_features"]
    seq_length = data_info["seq_length"]
    n_classes = data_info["n_classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Dataset info: features={n_features}, seq_length={seq_length}, classes={n_classes}")
    print(f"Device: {device}")

    # Generate hyperparameter combinations
    hyperparam_keys = list(RNN_HYPERPARAMS.keys())
    hyperparam_values = list(RNN_HYPERPARAMS.values())
    combinations = list(product(*hyperparam_values))

    results = []

    for idx, combo in enumerate(combinations, 1):
        hyperparams = dict(zip(hyperparam_keys, combo))
        print(f"\n[{idx}/{len(combinations)}] Testing RNN with: {hyperparams}")

        # Create model
        model = RNNClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            hidden_size=hyperparams['hidden_size'],
            n_layers=hyperparams['n_layers'],
            use_gru=hyperparams['use_gru'],
            bidirectional=True,
            dropout=0.1
        )

        # Benchmark
        result = benchmark_classification(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=f"RNN_{hyperparams}",
            dataset_name=dataset_name,
            n_epochs=n_epochs,
            device=device
        )

        # Store results with hyperparameters
        result_dict = {
            'model': 'RNN',
            'dataset': dataset_name,
            'hyperparams': str(hyperparams),
            'hidden_size': hyperparams['hidden_size'],
            'n_layers': hyperparams['n_layers'],
            'use_gru': hyperparams['use_gru'],
            'accuracy': result.metrics['accuracy'],
            'precision': result.metrics['precision'],
            'recall': result.metrics['recall'],
            'f1_macro': result.metrics['f1_macro'],
            'f1_weighted': result.metrics['f1_weighted'],
            'train_time': result.train_time,
            'inference_time': result.inference_time,
            'n_parameters': result.n_parameters,
            'threshold': None,
            'epoch_history': result.epoch_history
        }

        results.append(result_dict)
        print(f"Accuracy: {result.metrics['accuracy']:.4f}, F1: {result.metrics['f1_macro']:.4f}")

    return results


def run_cnn_experiment(dataset_name: str, n_epochs: int = 50) -> List[Dict[str, Any]]:
    """
    Run CNN classification experiment with hyperparameter variations.

    Args:
        dataset_name: Name of dataset (FordA or ECG200)
        n_epochs: Number of training epochs

    Returns:
        List of result dictionaries with metrics and hyperparameters
    """
    print(f"\n{'='*80}")
    print(f"Running CNN Experiment on {dataset_name}")
    print(f"{'='*80}")

    # Load dataset
    train_loader, test_loader, data_info = create_classification_loaders(
        dataset_name, batch_size=32
    )

    n_features = data_info["n_features"]
    seq_length = data_info["seq_length"]
    n_classes = data_info["n_classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Dataset info: features={n_features}, seq_length={seq_length}, classes={n_classes}")
    print(f"Device: {device}")

    # Generate hyperparameter combinations
    hyperparam_keys = list(CNN_HYPERPARAMS.keys())
    hyperparam_values = list(CNN_HYPERPARAMS.values())
    combinations = list(product(*hyperparam_values))

    results = []

    for idx, combo in enumerate(combinations, 1):
        hyperparams = dict(zip(hyperparam_keys, combo))
        print(f"\n[{idx}/{len(combinations)}] Testing CNN with: {hyperparams}")

        # Create model
        model = CNNClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            n_filters=hyperparams['n_filters'],
            n_blocks=hyperparams['n_blocks'],
            kernel_size=hyperparams['kernel_size'],
            dropout=0.1
        )

        # Benchmark
        result = benchmark_classification(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=f"CNN_{hyperparams}",
            dataset_name=dataset_name,
            n_epochs=n_epochs,
            device=device
        )

        # Store results with hyperparameters
        result_dict = {
            'model': 'CNN',
            'dataset': dataset_name,
            'hyperparams': str(hyperparams),
            'n_filters': hyperparams['n_filters'],
            'n_blocks': hyperparams['n_blocks'],
            'kernel_size': hyperparams['kernel_size'],
            'accuracy': result.metrics['accuracy'],
            'precision': result.metrics['precision'],
            'recall': result.metrics['recall'],
            'f1_macro': result.metrics['f1_macro'],
            'f1_weighted': result.metrics['f1_weighted'],
            'train_time': result.train_time,
            'inference_time': result.inference_time,
            'n_parameters': result.n_parameters,
            'threshold': None,
            'epoch_history': result.epoch_history
        }

        results.append(result_dict)
        print(f"Accuracy: {result.metrics['accuracy']:.4f}, F1: {result.metrics['f1_macro']:.4f}")

    return results


def run_vae_experiment(dataset_name: str, n_epochs: int = 50) -> List[Dict[str, Any]]:
    """
    Run VAE classification experiment with hyperparameter variations.

    Args:
        dataset_name: Name of dataset (FordA or ECG200)
        n_epochs: Number of training epochs

    Returns:
        List of result dictionaries with metrics and hyperparameters
    """
    print(f"\n{'='*80}")
    print(f"Running VAE Experiment on {dataset_name}")
    print(f"{'='*80}")

    # Load dataset
    train_loader, test_loader, data_info = create_classification_loaders(
        dataset_name, batch_size=32
    )

    n_features = data_info["n_features"]
    seq_length = data_info["seq_length"]
    n_classes = data_info["n_classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Dataset info: features={n_features}, seq_length={seq_length}, classes={n_classes}")
    print(f"Device: {device}")

    # Generate hyperparameter combinations
    hyperparam_keys = list(VAE_HYPERPARAMS.keys())
    hyperparam_values = list(VAE_HYPERPARAMS.values())
    combinations = list(product(*hyperparam_values))

    results = []

    for idx, combo in enumerate(combinations, 1):
        hyperparams = dict(zip(hyperparam_keys, combo))
        print(f"\n[{idx}/{len(combinations)}] Testing VAE with: {hyperparams}")

        # Create model
        model = VAEClassifier(
            n_features=n_features,
            seq_length=seq_length,
            n_classes=n_classes,
            hidden_size=hyperparams['hidden_size'],
            latent_dim=hyperparams['latent_dim'],
            n_layers=hyperparams['n_layers'],
            dropout=0.1
        )

        # Benchmark
        result = benchmark_classification(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=f"VAE_{hyperparams}",
            dataset_name=dataset_name,
            n_epochs=n_epochs,
            device=device
        )

        # Store results with hyperparameters
        result_dict = {
            'model': 'VAE',
            'dataset': dataset_name,
            'hyperparams': str(hyperparams),
            'hidden_size': hyperparams['hidden_size'],
            'latent_dim': hyperparams['latent_dim'],
            'n_layers': hyperparams['n_layers'],
            'accuracy': result.metrics['accuracy'],
            'precision': result.metrics['precision'],
            'recall': result.metrics['recall'],
            'f1_macro': result.metrics['f1_macro'],
            'f1_weighted': result.metrics['f1_weighted'],
            'train_time': result.train_time,
            'inference_time': result.inference_time,
            'n_parameters': result.n_parameters,
            'threshold': None,
            'epoch_history': result.epoch_history
        }

        results.append(result_dict)
        print(f"Accuracy: {result.metrics['accuracy']:.4f}, F1: {result.metrics['f1_macro']:.4f}")

    return results


# ============================================================================
# Results Aggregation and Table Generation
# ============================================================================

def create_results_dataframe(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create comprehensive DataFrame from all results.

    Args:
        all_results: List of result dictionaries

    Returns:
        DataFrame with all results
    """
    df = pd.DataFrame(all_results)

    # Reorder columns for better readability
    column_order = [
        'model', 'dataset', 'hyperparams', 'accuracy', 'precision', 'recall',
        'f1_macro', 'f1_weighted', 'train_time', 'inference_time',
        'n_parameters', 'threshold'
    ]

    # Add any hyperparameter columns that exist
    hyperparam_cols = [col for col in df.columns if col not in column_order and col != 'epoch_history']
    column_order_with_hyperparams = column_order[:3] + hyperparam_cols + column_order[3:]

    # Reorder keeping only existing columns
    final_columns = [col for col in column_order_with_hyperparams if col in df.columns]
    df = df[final_columns + ['epoch_history']]

    return df


def generate_summary_tables(df: pd.DataFrame):
    """
    Generate and save summary tables.

    Args:
        df: Results DataFrame
    """
    print("\n" + "="*80)
    print("SUMMARY TABLES")
    print("="*80)

    # Overall summary
    summary_cols = ['model', 'dataset', 'accuracy', 'precision', 'recall',
                   'f1_macro', 'train_time', 'inference_time', 'n_parameters']

    # Per-dataset tables
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset][summary_cols].copy()
        dataset_df = dataset_df.sort_values('accuracy', ascending=False)

        print(f"\n{dataset} Results:")
        print(dataset_df.to_string(index=False))

        # Save to CSV
        dataset_df.to_csv(RESULTS_DIR / f"{dataset}_results.csv", index=False)

    # Cross-dataset comparison (best config per model)
    print("\n" + "="*80)
    print("CROSS-DATASET COMPARISON (Best Configuration per Model)")
    print("="*80)

    best_configs = df.loc[df.groupby(['model', 'dataset'])['accuracy'].idxmax()]
    best_configs = best_configs[summary_cols].sort_values(['model', 'dataset'])

    print(best_configs.to_string(index=False))
    best_configs.to_csv(RESULTS_DIR / "best_configurations.csv", index=False)

    # Statistical summary by model across datasets
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY BY MODEL")
    print("="*80)

    stats_summary = df.groupby('model')[['accuracy', 'f1_macro', 'train_time', 'inference_time']].agg(['mean', 'std', 'min', 'max'])
    print(stats_summary)
    stats_summary.to_csv(RESULTS_DIR / "statistical_summary.csv")

    # Top 5 configurations overall
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS (All Models & Datasets)")
    print("="*80)

    top5 = df.nlargest(5, 'accuracy')[['model', 'dataset', 'hyperparams', 'accuracy', 'f1_macro', 'train_time']]
    print(top5.to_string(index=False))
    top5.to_csv(RESULTS_DIR / "top5_configurations.csv", index=False)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_loss_curves(df: pd.DataFrame):
    """Plot training loss curves for all configurations."""
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Loss Curves - {dataset}', fontsize=16, fontweight='bold')

        models = ['LSTM', 'RNN', 'CNN', 'VAE']
        for idx, model_name in enumerate(models):
            ax = axes[idx // 2, idx % 2]
            model_results = dataset_df[dataset_df['model'] == model_name]

            for _, row in model_results.iterrows():
                if row['epoch_history']:
                    epochs = [e['epoch'] for e in row['epoch_history']]
                    losses = [e['train_loss'] for e in row['epoch_history']]
                    ax.plot(epochs, losses, alpha=0.6, label=f"{row['hyperparams'][:30]}...")

            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Training Loss', fontsize=10)
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc='best')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'loss_curves_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved loss curves for {dataset}")


def plot_performance_metrics(df: pd.DataFrame):
    """
    Plot 2: Performance Metrics (Accuracy, Precision, Recall, F1).
    Combines accuracy comparison and metrics heatmap.
    """
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Performance Metrics Analysis', fontsize=16, fontweight='bold')

    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Best accuracy per model per dataset (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    best_acc = df.loc[df.groupby(['model', 'dataset'])['accuracy'].idxmax()]

    x = np.arange(len(best_acc['model'].unique()))
    width = 0.35
    datasets = best_acc['dataset'].unique()

    for i, dataset in enumerate(datasets):
        dataset_data = best_acc[best_acc['dataset'] == dataset]
        offset = width * (i - len(datasets)/2 + 0.5)
        ax1.bar(x + offset, dataset_data['accuracy'], width, label=dataset, alpha=0.8)

    ax1.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Best Accuracy', fontsize=10, fontweight='bold')
    ax1.set_title('Best Accuracy per Model', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(best_acc['model'].unique())
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Average accuracy per model (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    avg_acc = df.groupby(['model', 'dataset'])['accuracy'].mean().reset_index()

    for i, dataset in enumerate(datasets):
        dataset_data = avg_acc[avg_acc['dataset'] == dataset]
        offset = width * (i - len(datasets)/2 + 0.5)
        ax2.bar(x + offset, dataset_data['accuracy'], width, label=dataset, alpha=0.8)

    ax2.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Average Accuracy', fontsize=10, fontweight='bold')
    ax2.set_title('Average Accuracy per Model', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(avg_acc['model'].unique())
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Accuracy vs parameters scatter (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]
        ax3.scatter(dataset_data['n_parameters'], dataset_data['accuracy'],
                   label=dataset, alpha=0.6, s=50)

    ax3.set_xlabel('Number of Parameters', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
    ax3.set_title('Accuracy vs Model Complexity', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Metrics heatmap for all models (bottom - spans all columns)
    ax4 = fig.add_subplot(gs[1, :])

    # Create comprehensive metrics matrix
    metrics = ['accuracy', 'precision', 'recall', 'f1_macro']
    models = df['model'].unique()

    # Average metrics across datasets for each model
    heatmap_data = []
    for model in models:
        model_data = df[df['model'] == model]
        row = [model_data[metric].mean() for metric in metrics]
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
               xticklabels=[m.upper() for m in metrics],
               yticklabels=models,
               ax=ax4, cbar_kws={'label': 'Score'})

    ax4.set_title('Average Performance Metrics Across Datasets', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Metric', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Model', fontsize=10, fontweight='bold')

    plt.savefig(PLOTS_DIR / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved performance metrics plot")


def plot_time_analysis(df: pd.DataFrame):
    """
    Plot 3: Time Analysis (Training/Inference Time + Time vs Accuracy).
    Combines time comparison and time vs accuracy scatter plots.
    """
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Training & Inference Time Analysis', fontsize=16, fontweight='bold')

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    datasets = df['dataset'].unique()
    models = df['model'].unique()
    x = np.arange(len(models))
    width = 0.35

    # 1. Training time comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    train_data = df.groupby(['model', 'dataset'])['train_time'].mean().reset_index()

    for i, dataset in enumerate(datasets):
        dataset_data = train_data[train_data['dataset'] == dataset]
        offset = width * (i - len(datasets)/2 + 0.5)
        ax1.bar(x + offset, dataset_data['train_time'], width, label=dataset, alpha=0.8)

    ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Average Training Time', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Inference time comparison (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    inference_data = df.groupby(['model', 'dataset'])['inference_time'].mean().reset_index()

    for i, dataset in enumerate(datasets):
        dataset_data = inference_data[inference_data['dataset'] == dataset]
        offset = width * (i - len(datasets)/2 + 0.5)
        ax2.bar(x + offset, dataset_data['inference_time'], width, label=dataset, alpha=0.8)

    ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Inference Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Average Inference Time', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Train time vs accuracy scatter (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    for model, color in zip(models, colors):
        for dataset in datasets:
            data = df[(df['model'] == model) & (df['dataset'] == dataset)]
            marker = 'o' if dataset == datasets[0] else 's'
            ax3.scatter(data['train_time'], data['accuracy'],
                       c=[color], label=f'{model}-{dataset}',
                       alpha=0.6, s=60, marker=marker)

    ax3.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax3.set_title('Training Time vs Accuracy', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, loc='best', ncol=2)
    ax3.grid(True, alpha=0.3)

    # 4. Inference time vs accuracy scatter (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])

    for model, color in zip(models, colors):
        for dataset in datasets:
            data = df[(df['model'] == model) & (df['dataset'] == dataset)]
            marker = 'o' if dataset == datasets[0] else 's'
            ax4.scatter(data['inference_time'], data['accuracy'],
                       c=[color], label=f'{model}-{dataset}',
                       alpha=0.6, s=60, marker=marker)

    ax4.set_xlabel('Inference Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax4.set_title('Inference Time vs Accuracy', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=7, loc='best', ncol=2)
    ax4.grid(True, alpha=0.3)

    plt.savefig(PLOTS_DIR / 'time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved time analysis plot")


def plot_cross_dataset_comparison(df: pd.DataFrame):
    """
    Plot 4: Cross-Dataset Comparison.
    Best configurations compared across FordA and ECG200.
    """
    best_configs = df.loc[df.groupby(['model', 'dataset'])['accuracy'].idxmax()]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Cross-Dataset Performance Comparison (Best Configurations)',
                 fontsize=16, fontweight='bold')

    metrics = ['accuracy', 'f1_macro', 'train_time', 'inference_time']
    titles = ['Accuracy', 'F1 Score (Macro)', 'Training Time (s)', 'Inference Time (s)']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        pivot_data = best_configs.pivot(index='model', columns='dataset', values=metric)

        pivot_data.plot(kind='bar', ax=ax, alpha=0.8, width=0.7)
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(title='Dataset', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'cross_dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved cross-dataset comparison plot")


# ============================================================================
# Main Orchestration
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive time series classification experiments'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['FordA', 'ECG200'],
        help='Datasets to use (default: FordA ECG200)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['LSTM', 'RNN', 'CNN', 'VAE'],
        choices=['LSTM', 'RNN', 'CNN', 'VAE'],
        help='Models to run (default: all models)'
    )

    args = parser.parse_args()

    # Setup
    setup_directories()
    print(f"\nResults will be saved to: {RESULTS_DIR}")
    print(f"Plots will be saved to: {PLOTS_DIR}")

    # Run all experiments
    all_results = []

    for dataset in args.datasets:
        print(f"\n{'#'*80}")
        print(f"# Processing Dataset: {dataset}")
        print(f"{'#'*80}")

        # Run experiments for each model
        if 'LSTM' in args.models:
            lstm_results = run_lstm_experiment(dataset, args.epochs)
            all_results.extend(lstm_results)

        if 'RNN' in args.models:
            rnn_results = run_rnn_experiment(dataset, args.epochs)
            all_results.extend(rnn_results)

        if 'CNN' in args.models:
            cnn_results = run_cnn_experiment(dataset, args.epochs)
            all_results.extend(cnn_results)

        if 'VAE' in args.models:
            vae_results = run_vae_experiment(dataset, args.epochs)
            all_results.extend(vae_results)

    # Create comprehensive results DataFrame
    results_df = create_results_dataframe(all_results)

    # Save complete results
    results_df.drop('epoch_history', axis=1).to_csv(
        RESULTS_DIR / 'complete_results.csv', index=False
    )

    # Generate summary tables
    generate_summary_tables(results_df)

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_loss_curves(results_df)
    plot_performance_metrics(results_df)
    plot_time_analysis(results_df)
    plot_cross_dataset_comparison(results_df)

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Visualizations saved to: {PLOTS_DIR}")
    print("\nKey files generated:")
    print("\n  CSV Tables:")
    print("    - complete_results.csv: All experimental results")
    print("    - best_configurations.csv: Best config per model per dataset")
    print("    - statistical_summary.csv: Statistical analysis")
    print("    - top5_configurations.csv: Top 5 overall configurations")
    print("\n  Visualizations (4 plots):")
    print("    - loss_curves_FordA.png & loss_curves_ECG200.png: Training loss over epochs")
    print("    - performance_metrics.png: Accuracy, Precision, Recall, F1 analysis")
    print("    - time_analysis.png: Training/inference time comparison & efficiency")
    print("    - cross_dataset_comparison.png: FordA vs ECG200 performance")


if __name__ == "__main__":
    main()
