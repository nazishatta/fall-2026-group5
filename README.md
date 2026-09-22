# MNIST Baseline & NNSOM Analysis (v1_mnist)

> **GWU DATS 6501 Capstone | Group 5 | Advisor: Dr. Amir Jafari**  
> _Post-Training Analysis & Novelty Detection Using Neural-Network Self-Organizing Maps_

---

## Repository Layout

```
Code_local/
├── src/
│   └── v1_mnist/
│       ├── component/
│       │   ├── configs/
│       │   │   ├── base.yaml               # Shared hyperparameters
│       │   │   ├── cnn_baseline.yaml       # CNN baseline config
│       │   │   └── som.yaml                # NNSOM config
│       │   ├── data/
│       │   │   └── mnist_dataset.py        # CustomMNISTDataset + get_dataloaders()
│       │   ├── models/
│       │   │   └── lenet5.py               # LeNet-5 architecture
│       │   ├── features/
│       │   │   └── extractor.py            # FeatureExtractor (forward hooks)
│       │   ├── evaluation/
│       │   │   └── evaluator.py            # Accuracy, F1, confusion matrix
│       │   ├── visualization/
│       │   │   └── cnn_baseline_plots.py   # Training curves, confusion heatmap
│       │   ├── som/                        # NNSOM trainer, data loader, memory-safe init
│       │   ├── analysis/                   # SOM grid selection, BMU/RQ analysis
│       │   ├── utils/
│       │   │   └── config.py               # YAML loader with auto path-routing
│       │   └── pipeline/
│       │       ├── 01_prepare_data.py      # Split & validate MNIST
│       │       ├── 02_train_baseline.py    # Train LeNet-5
│       │       ├── 03_extract_embeddings.py# Extract fc2 embeddings
│       │       ├── 04_train_som.py         # Train SOM
│       │       └── 05_visualize_som.py     # SOM visualizations
│       ├── tests/
│       │   ├── conftest.py
│       │   ├── cnn_baseline/
│       │   │   ├── README.md
│       │   │   ├── test_data_loader.py     # Split sizes, image shape, pixel range
│       │   │   ├── test_lenet5.py          # Architecture, forward pass, numerics
│       │   │   └── test_feature_extractor.py # Shape (N,84), no NaN/Inf, alignment
│       │   └── som/
│       │       ├── test_bmu_rq_analysis.py # BMU/RQ analysis helpers
│       │       ├── test_memory_safe_init.py # SVD economy context
│       │       └── test_run_id_preflight.py # Output isolation preflight
│       ├── docs/
│       │   ├── cnn_baseline.md             # CNN technical documentation
│       │   └── som/
│       │       └── SOM_GRID_SELECTION_PROTOCOL.md # Grid selection protocol
│       └── shellscripts/
│           ├── README.md
│           └── run_cnn_baseline.sh         # End-to-end CNN baseline pipeline
├── cookbooks/
│   └── v1_mnist/
│       └── cnn_baseline.ipynb              # Baseline walkthrough notebook
├── demo/
│   └── fig/
│       └── v1_mnist/                       # Visualizations & figures
├── reports/
│   ├── Markdown_Report/
│   │   └── v1_mnist/
│   │       └── cnn_baseline_results.md
│   └── Progress_Report/
│       └── v1_mnist/
│           └── som_training_progress.md
└── outputs/
    └── v1_mnist/
        ├── cnn_baseline/                   # Checkpoints, embeddings, figures, logs
        └── som/                            # SOM models, logs, tables, figures
```

---

## CNN Baseline Results

| Metric              | Value                    |
| ------------------- | ------------------------ |
| Test Accuracy       | **99.18%**               |
| Macro F1-Score      | **0.9918**               |
| Embedding dimension | 84 (fc2 layer)           |
| Train / Val / Test  | 49,000 / 10,500 / 10,500 |

See [`reports/Markdown_Report/v1_mnist/cnn_baseline_results.md`](reports/Markdown_Report/v1_mnist/cnn_baseline_results.md)
and [`src/v1_mnist/docs/cnn_baseline.md`](src/v1_mnist/docs/cnn_baseline.md) for full details.

---

## Running Tests

From the project root (`Codes/Code_local`):

```bash
python -m pytest src/v1_mnist/tests -v
```
