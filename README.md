# Deep Learning for Time Series Classification

AI4DB Course Project: Comprehensive benchmarking of deep learning architectures for time series classification with hyperparameter optimization.

## Overview

This project benchmarks 4 neural network architectures on time series classification tasks with extensive hyperparameter search:

| Architecture | Description |
|--------------|-------------|
| LSTM | Long Short-Term Memory with bidirectional support and dropout |
| RNN | GRU-based recurrent network with bidirectional option |
| CNN | 1D convolutions with residual connections and batch normalization |
| VAE | Variational Autoencoder adapted for classification tasks |

## Task

**Classification**: Predict class labels from univariate time series (UCR Archive datasets)

The experiment performs comprehensive hyperparameter search:
- **4 models** × **2 datasets** × **4 hyperparameter configurations** = **32 total experiments**
- Automated generation of performance comparisons, statistical summaries, and visualizations

## Datasets

Datasets are excluded from version control via `.gitignore`. Download them before running experiments.

| Dataset | Source | Description | Train Size | Test Size | Sequence Length | Classes |
|---------|--------|-------------|------------|-----------|-----------------|---------|
| FordA | UCR Archive | Engine sensor classification | 3601 | 1320 | 500 | 2 |
| ECG200 | UCR Archive | ECG heartbeat classification | 100 | 100 | 96 | 2 |

## Project Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── requirements_visual.txt   # Dependencies for interactive demo
├── main.py                   # Main experiment orchestration with hyperparameter search
├── old_main.py               # Legacy experiment script
├── QUICKSTART_VISUAL.md      # Quick start guide for demo
├── README_VISUAL.md          # Full documentation for demo
├── data/                     # Downloaded datasets (gitignored)
│   └── ucr/
│       ├── FordA/
│       └── ECG200/
├── results/                  # Experiment results and visualizations
│   ├── *.csv                 # Performance metrics tables
│   └── plots/                # Training curves and comparison plots
├── checkpoints/              # Pre-trained models for demo (gitignored)
│   └── ecg200_best/
│       ├── lstm.pt
│       ├── rnn.pt
│       ├── cnn.pt
│       └── vae.pt
├── scripts/
│   └── train_demo_models.py  # Train models for interactive demo
└── src/
    ├── __init__.py
    ├── download_data.py      # Dataset download utilities
    ├── data_loader.py        # PyTorch data loaders
    ├── benchmark.py          # Training and evaluation utilities
    ├── plots.py              # Visualization utilities
    ├── models/
    │   ├── __init__.py
    │   ├── lstm.py           # LSTM classifier
    │   ├── cnn.py            # 1D CNN classifier
    │   ├── rnn.py            # GRU-based RNN classifier
    │   └── vae.py            # VAE classifier
    └── visual_apps/          # Interactive demo application
        ├── __init__.py
        ├── app.py            # Main Streamlit application
        ├── components/       # UI components
        │   ├── __init__.py
        │   └── prediction_viewer.py
        └── utils/            # Helper utilities
            ├── __init__.py
            └── model_loader.py
```

## Setup

### Option 1: Conda (recommended)

Create and activate a new conda environment:

```bash
conda create -n timeseries python=3.10
conda activate timeseries
```

Install PyTorch (with CUDA support if available):

```bash
# For GPU support (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

### Option 2: pip with venv

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start: Interactive Demo 🚀

Want to explore the models interactively? We have a web-based demo!

### 1. Install dependencies
```bash
pip install -r requirements_visual.txt
```

### 2. Download dataset (if not already done)
```bash
python -c "from src.download_data import download_all_datasets; download_all_datasets()"
```

### 3. Train demo models (~30 seconds)
```bash
python scripts/train_demo_models.py
```

This trains all 4 models: LSTM (77%), RNN (84%), CNN (82%), and VAE (76%).

### 4. Launch the interactive demo
```bash
streamlit run src/visual_apps/app.py
```

Your browser will open at `http://localhost:8501` with an interactive interface to:
- Explore predictions on individual ECG samples
- Compare all 4 model architectures (LSTM vs RNN vs CNN vs VAE)
- View model confidence scores and compare behaviors
- Visualize time series data interactively
- Understand differences between recurrent, convolutional, and variational models

---

## Usage (Command Line Benchmarking)

### Step 1: Download datasets

Datasets are not included in the repository. Download them first:

```bash
python -c "from src.download_data import download_ucr_dataset; download_ucr_dataset('FordA'); download_ucr_dataset('ECG200')"
```

### Step 2: Run experiments

Run all experiments with default settings (50 epochs, both datasets, all models):

```bash
python main.py --epochs 50
```

### Customize experiments

Run specific datasets:
```bash
python main.py --epochs 50 --datasets FordA
```

Run specific models:
```bash
python main.py --epochs 50 --models LSTM CNN
```

Run single dataset with subset of models:
```bash
python main.py --epochs 50 --datasets ECG200 --models LSTM RNN
```

## Hyperparameter Search

The main script performs grid search over the following hyperparameter spaces:

### LSTM
- `hidden_size`: [64, 128]
- `n_layers`: [1, 2]
- `bidirectional`: [False]
- **Total**: 4 configurations per dataset

### RNN (GRU)
- `hidden_size`: [64, 128]
- `n_layers`: [1, 2]
- `use_gru`: [True]
- **Total**: 4 configurations per dataset

### CNN
- `n_filters`: [32, 64]
- `n_blocks`: [2, 3]
- `kernel_size`: [3]
- **Total**: 4 configurations per dataset

### VAE
- `hidden_size`: [32, 64]
- `latent_dim`: [8, 16]
- `n_layers`: [1]
- **Total**: 4 configurations per dataset

## Model Details

### LSTM

Bidirectional LSTM classifier:
- Multi-layer stacked LSTM cells
- Optional bidirectional processing
- Dropout for regularization
- Final hidden state for classification

### RNN

GRU-based recurrent network:
- Multi-layer stacked GRU cells
- Bidirectional option enabled by default
- Simpler than LSTM with comparable performance
- Dropout for regularization

### CNN

1D convolutional network with residual connections:
- Batch normalization after each conv layer
- ReLU activation
- Multiple residual blocks
- Global average pooling for classification
- Dropout for regularization

### VAE

Variational Autoencoder adapted for classification:
- LSTM encoder and decoder
- Learns latent representation
- Classification head on latent features
- KL divergence regularization

## Metrics

All experiments report the following classification metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Macro-averaged precision
- **Recall**: Macro-averaged recall
- **F1 Score**: Both macro and weighted F1 scores
- **Training Time**: Total training time in seconds
- **Inference Time**: Test set inference time in seconds
- **Parameters**: Number of trainable parameters

## Results

Results are automatically saved to the `results/` directory:

### CSV Tables
- `complete_results.csv`: All experimental results with hyperparameters
- `{dataset}_results.csv`: Per-dataset performance summary
- `best_configurations.csv`: Best configuration per model per dataset
- `statistical_summary.csv`: Mean, std, min, max for each model
- `top5_configurations.csv`: Top 5 configurations overall

### Visualizations (PNG)
- `loss_curves_{dataset}.png`: Training loss over epochs for all configurations
- `performance_metrics.png`: Accuracy, precision, recall, F1 comparison
- `time_analysis.png`: Training/inference time vs accuracy tradeoffs
- `cross_dataset_comparison.png`: Performance comparison across FordA and ECG200

## Example Output

After running experiments, you'll see:

```
Results saved to: results/
Visualizations saved to: results/plots/

Key files generated:
  CSV Tables:
    - complete_results.csv
    - best_configurations.csv
    - statistical_summary.csv
    - top5_configurations.csv

  Visualizations:
    - loss_curves_FordA.png & loss_curves_ECG200.png
    - performance_metrics.png
    - time_analysis.png
    - cross_dataset_comparison.png
```

## Dependencies

- PyTorch >= 2.0
- NumPy >= 1.24
- Pandas >= 2.0
- Scikit-learn >= 1.3
- Matplotlib >= 3.7
- Seaborn >= 0.12
- SciPy >= 1.11
- tqdm >= 4.65
- requests >= 2.31

## References

- UCR Time Series Archive: https://www.timeseriesclassification.com/
- Hochreiter and Schmidhuber "Long Short-Term Memory" (1997)
- Cho et al. "Learning Phrase Representations using RNN Encoder-Decoder" (2014)
- Kingma and Welling "Auto-Encoding Variational Bayes" (2013)
