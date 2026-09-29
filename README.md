# MNIST Baseline & NNSOM Analysis (v1_mnist)

> **GWU DATS 6501 Capstone | Group 5 | Advisor: Dr. Amir Jafari**  
> _Post-Training Analysis & Novelty Detection Using Neural-Network Self-Organizing Maps_

---

## Why this matters

A model can post strong average accuracy while hiding structured weaknesses. Existing
interpretability tools — saliency maps, SHAP, LIME — explain one prediction at a time.
They answer *which pixels mattered for this sample*, not *where in the representation do
this model's failures concentrate*. That population-level view is what determines whether
a model is safe to deploy, and it is what this project tries to recover.

## Central research hypothesis

> SOM-based topology and quantization signals can expose structured regions of model
> failure in learned representation space; those signals can detect distributional
> novelty; and SOM-derived diagnostics can guide targeted interventions that measurably
> improve predictive behavior and/or the structure of the learned representation.

This is treated as a hypothesis to test, not a conclusion to prove. Components
unsupported by evidence will be reported as unsupported.

## Pipeline Architecture

<p align="center">
  <img src="demo/fig/project_pipeline/nnsom_project_pipeline.drawio.svg"
       alt="NNSOM Project Pipeline Architecture"
       width="100%">
</p>

## Research questions

| | Question | Primary output |
|---|---|---|
| **RQ1** | What does the latent feature space look like under a topology-preserving projection? | Cluster structure, class overlap, density maps |
| **RQ2** | Do misclassifications occupy identifiable regions of the representation? | Error hotspot statistics + representative samples |
| **RQ3** | Can SOM-derived measures identify out-of-distribution observations? | AUROC / AUPR / FPR@95TPR |
| **RQ4** | How does SOM novelty detection compare with established baselines? | Isolation Forest, One-Class SVM, autoencoder |
| **RQ5** | Can SOM-derived error regions identify samples worth targeting in retraining? | Guided vs. **random-selection control** |
| **RQ6** | After intervention, is the representation measurably better structured? | Before/after representation metrics |

Success criteria for each RQ are fixed **before** the test set is examined
(to be written in `reports/`, see the proposal).

## Key results

**CNN baseline (LeNet-5 on MNIST), measured on the held-out test split:**

| Metric              | Value                    |
| ------------------- | ------------------------ |
| Test accuracy       | **99.18%**               |
| Macro F1-score      | **0.9918**               |
| Embedding dimension | 84 (fc2 layer)           |
| Train / Val / Test  | 49,000 / 10,500 / 10,500 |

A 15x15 SOM has been trained on the fc2 embeddings (run `som_15x15_seed42_final`).
Results for RQ2-RQ6 (error geography, OOD detection, SOM-guided retraining) are
`[TBD]` until the experiments are run. Nothing is reported before it is measured.

Details: [`reports/Markdown_Report/v1_mnist/cnn_baseline_results.md`](reports/Markdown_Report/v1_mnist/cnn_baseline_results.md),
[`reports/Markdown_Report/v1_mnist/som_15x15_visualizations.md`](reports/Markdown_Report/v1_mnist/som_15x15_visualizations.md)
and [`src/v1_mnist/docs/cnn_baseline.md`](src/v1_mnist/docs/cnn_baseline.md).

## Repository structure

```
src/v1_mnist/
  component/
    configs/         YAML settings (base, cnn_baseline, som, dataset/)
    data/            MNIST dataset and dataloaders
    models/          LeNet-5
    features/        fc2 embedding extraction
    som/             NNSOM training, data loading, memory-safe init
    analysis/        BMU / RQ1 analysis (io, metrics, plotting, rq1)
    evaluation/      accuracy, F1, confusion matrix
    visualization/   CNN and SOM figures (SVG + PDF)
    interactive/     interactive SOM HTML report
    pipeline/        runner.py plus one script per stage
    ood/, retraining/  planned (RQ3-RQ6)
    utils/           config loading, logging
  docs/              technical notes and SOM grid-selection protocol
  shellscripts/      run scripts
  tests/             pytest tests
cookbooks/           walkthrough notebook
demo/                figures and interactive plots used in reports
outputs/v1_mnist/    metrics, tables, logs and figures (large arrays are not stored in git)
reports/             Markdown, Word and LaTeX reports, progress reports, proposal
presentation/        slides
research_paper/      paper drafts
```

Figures are saved as vector files (SVG and PDF).

## Installation

```bash
git clone https://github.com/nazishatta/fall-2026-group5.git
cd fall-2026-group5
python -m venv .venv
source .venv/bin/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

Run from the project root.

```bash
# Run the test suite
python -m pytest src/v1_mnist/tests -q

# Run all six stages (data -> baseline -> embeddings -> SOM -> figures -> interactive report)
python src/v1_mnist/component/pipeline/runner.py --stage all --run-id som_15x15_seed42_final
```

To run one stage, use for example `--stage train_som`. Stage order and options are in
[`src/v1_mnist/component/pipeline/README.md`](src/v1_mnist/component/pipeline/README.md).

## Reproduction

Every reported number traces back through
**config -> raw result -> processed metric -> table/figure -> conclusion**.
Experiment names encode their configuration, e.g.
`fashion_mnist_lenet5_penultimate_som15x15_seed42`.

## Data

MNIST downloads automatically via torchvision on first run. Fashion-MNIST and CIFAR-10
are planned. Embedding `.npy` files are not stored in git; they will be tracked with DVC
(S3 remote access is pending) and can otherwise be regenerated with the pipeline.

## Technologies

PyTorch, [NNSOM](https://amir-jafari.github.io/SOM/), scikit-learn, NumPy, Matplotlib, Plotly, pytest, DVC

## Limitations

`[TBD - written from actual findings, not anticipated ones]`

## Team

- **Fyrooz Khan** - CNN baseline, NNSOM training and visualization
- **Nazish Atta** - project setup, documentation and cluster analysis

Advisor: Dr. Amir Jafari, The George Washington University, Data Science Program.
Developed as a capstone project in the GWU MS Data Science program.
