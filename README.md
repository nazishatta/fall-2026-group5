# MNIST Baseline & NNSOM Analysis (v1_mnist)

> **GWU DATS 6501 Capstone | Group 5 | Advisor: Dr. Amir Jafari**  
> _Post-Training Analysis & Novelty Detection Using Neural-Network Self-Organizing Maps_

---

## Why this matters

A model can post strong average accuracy while hiding structured weaknesses. Existing
interpretability tools — saliency maps, SHAP, LIME — explain one prediction at a time.
They answer _which pixels mattered for this sample_, not _where in the representation do
this model's failures concentrate_. That population-level view is what determines whether
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

## Project phases

The phases and weeks follow the project proposal ([`reports/Project_proposal.md`](reports/Project_proposal.md)).

| Phase | Weeks | What is done | Main outputs |
| --- | --- | --- | --- |
| **1: Model Training Baseline** | 1–2 | Data pipeline; train a baseline CNN; extract feature embeddings for every split | Accuracy, precision, recall, F1; 84-D fc2 embeddings |
| **2: SOM Training & Visualization** | 3–5 | Week 3: train the SOM with NNSOM. Week 4: Cluster Analysis. Week 5: Error Pattern Analysis | SOM maps, cluster purity, class–cluster intersection, error hotspots and their images |
| **3: Novelty Detection** | 6–8 | Week 6: novelty detection via quantization error. Week 7: U-Matrix analysis. Week 8: Comparative Evaluation | AUROC, AUPR, FPR at 95% TPR against Isolation Forest, One-Class SVM and an autoencoder |
| **4: Guided Retraining** | 9–11 | Week 9: hard example mining. Week 10: targeted data augmentation. Week 11: iterative SOM re-analysis | Retrained model vs. baseline; old vs. new SOM maps |
| **5: Paper Writing & Code Release** | 12–14 | Final results and figures, paper draft, code release | Paper, documented code |

Success criteria are fixed **before** the test set is examined
(to be written in `reports/`).

## Key results

**CNN baseline (LeNet-5 on MNIST), measured on the held-out test split:**

| Metric              | Value                    |
| ------------------- | ------------------------ |
| Test accuracy       | **99.18%**               |
| Macro F1-score      | **0.9918**               |
| Embedding dimension | 84 (fc2 layer)           |
| Train / Val / Test  | 49,000 / 10,500 / 10,500 |

A 15x15 SOM has been trained on the fc2 embeddings (official model `som_15x15_nb11_ep250_s42_v2`).
Why 15x15, and how the grid choice was checked: [`src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/README.md`](src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/README.md).

**Week 5, Error Pattern Analysis (error geography):** CNN errors concentrate in a few hotspot cells (about 11 times their share of images)
and in cells where digit classes mix. Both findings also hold on 10x10 and 20x20 SOMs
([`ERROR_PATTERN_ROBUSTNESS_RESULTS.md`](src/v1_mnist/docs/p2_som_training_visualization/error_pattern_analysis/ERROR_PATTERN_ROBUSTNESS_RESULTS.md)).
Results for Phases 3–4 (novelty detection, guided retraining) are `[TBD]` until the experiments are run.
Nothing is reported before it is measured.

Details: [`reports/Markdown_Report/v1_mnist/cnn_baseline_results.md`](reports/Markdown_Report/v1_mnist/cnn_baseline_results.md),
[`reports/Markdown_Report/v1_mnist/som_15x15_visualizations.md`](reports/Markdown_Report/v1_mnist/som_15x15_visualizations.md)
and [`src/v1_mnist/docs/p1_model_training_baseline/cnn_baseline_training/cnn_baseline.md`](src/v1_mnist/docs/p1_model_training_baseline/cnn_baseline_training/cnn_baseline.md).

## Repository structure

```
src/v1_mnist/
  component/                         code, by project phase and week (p = phase)
    configs/                         YAML settings (base, cnn_baseline, som, dataset/)
    pipeline/                        one runner per week + run_all.py
    p1_model_training_baseline/
      data_pipeline/                 Week 1: MNIST split, class distributions
      cnn_baseline_training/         Week 2: LeNet-5, evaluation, fc2 embeddings
    p2_som_training_visualization/
      som_nnsom_training/            Week 3: NNSOM training, quality metrics, SOM figures
      cluster_analysis/              Week 4: BMUs, purity, cluster maps
      error_pattern_analysis/        Week 5: error hotspots, robustness check
      interactive_report/            Weeks 3-5: interactive SOM HTML report
    p3_novelty_detection/            Weeks 6-8 (planned)
    p4_guided_retraining/            Weeks 9-11 (planned)
    utils/                           config, logging, run settings, shared I/O, stats, heatmaps
  docs/                              technical notes and SOM grid-selection protocol
  shellscripts/                      run scripts
  tests/                             pytest tests, same phase/week folders as component/
cookbooks/           walkthrough notebook
demo/                figures and interactive plots used in reports
outputs/v1_mnist/    results, in the same phase/week folders (git keeps only small .json/.csv
                     metrics and .dvc pointers; data is tracked with DVC, figures are regenerated)
reports/             Markdown, Word and LaTeX reports, progress reports, proposal
presentation/        slides
research_paper/      paper drafts
```

Figures are saved as PDF only, with fonts embedded as TrueType (`pdf.fonttype` 42), by `utils/figures.py`.

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

# See every step without running anything
python src/v1_mnist/component/pipeline/run_all.py --from data_pipeline --dry-run

# Redraw the SOM figures from the official 15x15 SOM, then run the cluster and
# error-pattern analyses and the report (a step whose results already exist stops
# the run; results are never overwritten, so move or delete them to redo a step)
python src/v1_mnist/component/pipeline/run_all.py --skip-training
```

There is one runner per proposal step in `src/v1_mnist/component/pipeline/`. The `data_pipeline` and
`cnn_baseline_training` runners refuse to run while their results exist: running them again would replace
the frozen embeddings and change every result.
Which runners to use, and which not to: [`src/v1_mnist/component/pipeline/README.md`](src/v1_mnist/component/pipeline/README.md).

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

- **Fyrooz Khan** - Project set up, CNN baseline, NNSOM training and visualization, documentation
- **Nazish Atta** - NNSOM training, cluster analysis, error analysis and documentation

Advisor: Dr. Amir Jafari, The George Washington University, Data Science Program.
Developed as a capstone project in the GWU MS Data Science program.
