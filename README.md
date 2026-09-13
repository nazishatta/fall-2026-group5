# MNIST Baseline Pipeline

This project contains a PyTorch pipeline for preparing the MNIST dataset, training a LeNet-5 baseline CNN, and extracting penultimate-layer embeddings.

The repository follows the required Capstone project folder structure, with project source code organized under `src/component/`.

## Directory Structure

```text
.
├── cookbooks/
│
├── demo/
│   └── fig/
│
├── outputs/
│   ├── embeddings/
│   └── figures/
│
├── presentation/
│
├── reports/
│   ├── Latex_report/
│   │   └── fig/
│   ├── Markdown_Report/
│   ├── Progress_Report/
│   │   └── Markdown_CheatSheet/
│   └── Word_Report/
│
├── research_paper/
│   ├── Latex/
│   │   └── Fig/
│   └── Word/
│
├── src/
│   ├── component/
│   │   ├── analysis/
│   │   ├── configs/
│   │   │   ├── base.yaml
│   │   │   ├── mnist_config.yaml
│   │   │   └── dataset/
│   │   │       └── mnist.yaml
│   │   │
│   │   ├── data/
│   │   │   └── mnist_dataset.py
│   │   │
│   │   ├── evaluation/
│   │   │   └── metrics.py
│   │   │
│   │   ├── features/
│   │   │   └── extractor.py
│   │   │
│   │   ├── models/
│   │   │   └── lenet5.py
│   │   │
│   │   ├── ood/
│   │   ├── pipeline/
│   │   │   ├── 01_prepare_data.py
│   │   │   ├── 02_train_baseline.py
│   │   │   └── 03_extract_embeddings.py
│   │   │
│   │   ├── retraining/
│   │   ├── som/
│   │   ├── utils/
│   │   │   └── config.py
│   │   └── visualization/
│   │       └── plots.py
│   │
│   ├── docs/
│   ├── shellscripts/
│   └── tests/
│
├── .gitignore
├── README.md
└── requirements.txt
```
