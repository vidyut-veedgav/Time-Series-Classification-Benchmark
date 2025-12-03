# Deep Learning for Time Series

AI4DB Course Project: Comparing deep learning architectures for time series analysis.

## Overview

This project benchmarks neural network architectures on time series tasks:

| Architecture | Description |
|--------------|-------------|
| Transformer | Self-attention based encoder with positional encoding |
| LSTM | Long Short-Term Memory with bidirectional support |
| CNN | 1D convolutions with residual connections |
| RNN | GRU-based recurrent network |
| VAE | Variational Autoencoder for anomaly detection |
| STL | STL decomposition with neural network components |
| ImageCNN | 2D CNN on spectrogram/GAF/recurrence plot representations |

## Tasks

1. Classification: Predict class labels from time series (UCR datasets)
2. Forecasting: Predict future values (ETT dataset)
3. Anomaly Detection: Identify anomalous patterns (synthetic data)
4. Medical: ECG arrhythmia classification (MIT-BIH Arrhythmia Database)

## Datasets

Datasets and model weights are excluded from version control via `.gitignore`. Each user should download data locally.

| Dataset | Task | Source | Description |
|---------|------|--------|-------------|
| ECG200 | Classification | UCR Archive | ECG heartbeat classification |
| ECG5000 | Classification | UCR Archive | Larger ECG dataset |
| FordA | Classification | UCR Archive | Engine sensor classification |
| ETTh1/ETTm1 | Forecasting | ETDataset | Electricity transformer temperature |
| MIT-BIH | Classification | PhysioNet | 3-class arrhythmia (Normal, Supraventricular, Ventricular) |
| Synthetic | Anomaly | Generated | Synthetic anomaly detection data |

## Project Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── data/                 # Downloaded datasets (gitignored)
│   ├── ucr/
│   ├── ett/
│   ├── medical/
│   └── anomaly/
├── weights/              # Model checkpoints (gitignored)
├── results/
└── src/
    ├── __init__.py
    ├── download_data.py
    ├── data_loader.py
    ├── benchmark.py
    ├── plots.py
    └── models/
        ├── __init__.py
        ├── transformer.py
        ├── lstm.py
        ├── cnn.py
        ├── rnn.py
        ├── vae.py
        ├── stl.py
        └── image_cnn.py
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

## Usage

### Step 1: Download datasets

Datasets are not included in the repository. Run this first to download all required data:

```bash
python -c "from src.download_data import download_all_datasets; download_all_datasets()"
```

Or download and run experiments in one command:

```bash
python main.py --download --task all
```

### Step 2: Run experiments

```bash
python main.py --task all --epochs 50
```

### Run specific tasks

Classification:
```bash
python main.py --task classification --dataset ECG200 --epochs 50
```

Forecasting:
```bash
python main.py --task forecasting --dataset ETTh1.csv --epochs 50
```

Anomaly detection:
```bash
python main.py --task anomaly --epochs 50
```

Medical ECG classification:
```bash
python main.py --task medical --epochs 50
```

## Model Details

### Transformer

Based on the original architecture with modifications for time series:
- Positional encoding for temporal information
- Mean pooling over sequence for classification
- Encoder-decoder architecture for forecasting

### LSTM

Bidirectional LSTM with attention mechanism:
- Multi-layer stacked LSTM cells
- Attention for forecasting tasks
- Final hidden state for classification

### CNN

1D convolutional network with residual connections:
- Batch normalization
- Dilated convolutions for large receptive fields
- Global average pooling for classification

### RNN

GRU-based recurrent network:
- Bidirectional option for classification
- Encoder-decoder for forecasting
- Simpler than LSTM with comparable performance

### VAE

Variational Autoencoder for unsupervised anomaly detection:
- LSTM encoder and decoder
- Reconstruction error as anomaly score
- Threshold-based detection

### STL (Seasonal-Trend Decomposition)

Hybrid classical/neural approach:
- Decomposes signal into trend, seasonal, and residual components
- Separate neural networks model each component
- Combines predictions for final output

### ImageCNN

Treats 1D time series as 2D images:
- Converts signals to spectrograms (STFT-based)
- Gramian Angular Field (GAF) transformation
- Recurrence plot representation
- Standard 2D CNN for classification
- MultiScale variant combines all three representations

## Metrics

Classification:
- Accuracy
- F1 Score (macro and weighted)
- Precision
- Recall

Forecasting:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R2 Score

Anomaly Detection:
- Precision
- Recall
- F1 Score

## Results

Results are saved to the `results/` directory:
- JSON files with metrics and training history
- PNG plots for training curves and comparisons

## Dependencies

- PyTorch >= 2.0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy
- tqdm
- wfdb (for PhysioNet data access)

## References

- UCR Time Series Archive: https://www.timeseriesclassification.com/
- ETDataset: https://github.com/zhouhaoyi/ETDataset
- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/
- Vaswani et al. "Attention Is All You Need" (2017)
- Hochreiter and Schmidhuber "Long Short-Term Memory" (1997)
- Kingma and Welling "Auto-Encoding Variational Bayes" (2013)
- Cleveland et al. "STL: A Seasonal-Trend Decomposition" (1990)
- Wang and Oates "Imaging Time-Series to Improve Classification" (2015)
