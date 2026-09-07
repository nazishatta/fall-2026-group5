# MNIST Baseline Pipeline

This project contains a PyTorch pipeline for preparing the MNIST dataset, training a LeNet-5 baseline CNN, and extracting penultimate layer embeddings.

## Directory Structure
```
.
├── configs/
│   └── mnist_config.yaml        # Configuration file for paths, hyperparameters
├── scripts/
│   ├── 01_prepare_data.py       # Data downloading and 70/15/15 splitting
│   ├── 02_train_baseline.py     # LeNet-5 model training
│   └── 03_extract_embeddings.py # Feature extraction from penultimate layer
├── src/
│   ├── data/                    # Dataset handling
│   ├── evaluation/              # Metrics and testing
│   ├── features/                # Feature extraction
│   ├── models/                  # CNN architectures
│   ├── utils/                   # Config, seeds, devices
│   └── visualization/           # Plotting utilities
└── README.md
```

## Dataset Splitting
By default, torchvision's MNIST provides a 60k/10k train/test split. This project combines all 70,000 samples and performs a stratified split into:
- 70% Training (49,000 samples)
- 15% Validation (10,500 samples)
- 15% Test (10,500 samples)

## Usage

Run the scripts in the following order from this directory on the Ubuntu VM:

### 1. Data Preparation
Downloads/loads the dataset, performs the stratified splits, and generates distribution plots.
```bash
python scripts/01_prepare_data.py --config configs/mnist_config.yaml
```

### 2. Model Training
Trains the LeNet-5 model, evaluates on the test set, and plots learning curves.
```bash
python scripts/02_train_baseline.py --config configs/mnist_config.yaml
```

### 3. Feature Extraction
Extracts the 84-dimensional features from the `fc2` layer using the best saved checkpoint.
```bash
python scripts/03_extract_embeddings.py --config configs/mnist_config.yaml --checkpoint outputs/checkpoints/lenet5_best.pth
```

## Outputs
Outputs will be generated in the `outputs/` folder (configured in `mnist_config.yaml`):
- `outputs/figures/`: Class distributions, sample images, training curves, confusion matrix.
- `outputs/checkpoints/`: Saved model weights (`lenet5_best.pth`).
- `outputs/embeddings/mnist/`: Saved `.npy` files for features, labels, and predictions.
