# MNIST Baseline – Post-Training Analysis with NNSOM

> **GWU DATS 6501 Capstone | Group 5 | Advisor: Dr. Amir Jafari**  
> _Post-Training Analysis & Novelty Detection Using Neural-Network Self-Organizing Maps_

---

## Repository Layout

```
week_2_codes/
├── src/
│   ├── component/
│   │   ├── configs/
│   │   │   ├── base.yaml               # Shared hyperparameters (all weeks)
│   │   │   ├── week_2_baseline.yaml    # Week 2 milestone config
│   │   │   └── week_3_som.yaml         # Week 3 NNSOM config
│   │   ├── data/
│   │   │   └── mnist_dataset.py        # CustomMNISTDataset + get_dataloaders()
│   │   ├── models/
│   │   │   └── lenet5.py               # LeNet-5 architecture
│   │   ├── features/
│   │   │   └── extractor.py            # FeatureExtractor (forward hooks)
│   │   ├── evaluation/
│   │   │   └── evaluator.py            # Accuracy, F1, confusion matrix
│   │   ├── visualization/
│   │   │   └── plotter.py              # Training curves, confusion heatmap
│   │   ├── utils/
│   │   │   └── config.py               # YAML loader with auto path-routing
│   │   └── pipeline/
│   │       ├── 01_prepare_data.py      # Split & validate MNIST
│   │       ├── 02_train_baseline.py    # Train LeNet-5
│   │       └── 03_extract_embeddings.py# Extract fc2 embeddings
│   ├── tests/
│   │   ├── conftest.py
│   │   └── week_2/
│   │       ├── README.md
│   │       ├── test_data_loader.py         # Split sizes, image shape, pixel range
│   │       ├── test_lenet5.py              # Architecture, forward pass, numerics
│   │       └── test_feature_extractor.py  # Shape (N,84), no NaN/Inf, alignment
│   ├── docs/
│   │   └── week_2_baseline.md          # Technical pipeline documentation
│   └── shellscripts/
│       ├── README.md
│       └── run_week_2_baseline.sh      # One-command full pipeline
├── reports/
│   └── Progress_Report/
│       └── week_2_progress.md          # Week 2 results & decisions
└── outputs/                            # Auto-created; NOT committed to git
    └── week_2/
        ├── checkpoints/
        ├── embeddings/
        └── figures/
```

---

## Week 2 Results

| Metric              | Value                    |
| ------------------- | ------------------------ |
| Test Accuracy       | **99.18%**               |
| Macro F1-Score      | **0.9918**               |
| Embedding dimension | 84 (fc2 layer)           |
| Train / Val / Test  | 49,000 / 10,500 / 10,500 |

See [`reports/Progress_Report/week_2_progress.md`](reports/Progress_Report/week_2_progress.md)
and [`src/docs/week_2_baseline.md`](src/docs/week_2_baseline.md) for full details.

---

## Adding a New Week

1. Create `src/component/configs/week_N_<name>.yaml` with `run.name: week_N`.
2. Add any new hyperparameters (all shared ones are inherited from `base.yaml`).
3. New pipeline scripts auto-save to `outputs/week_N/`.

No other files need to change.

---
