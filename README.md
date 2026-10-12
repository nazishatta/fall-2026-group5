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
  <img src="demo/project_pipeline/nnsom_project_pipeline.drawio.svg"
       alt="NNSOM Project Pipeline Architecture"
       width="100%">
</p>

## Project phases

The phases and weeks follow the project proposal ([`reports/Project_proposal.md`](reports/Project_proposal.md)).

| Phase                               | Weeks | What is done                                                                                                | Main outputs                                                                           |
| ----------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **1: Model Training Baseline**      | 1–2   | Data pipeline; train a baseline CNN; extract feature embeddings for every split                             | Accuracy, precision, recall, F1; 84-D fc2 embeddings                                   |
| **2: SOM Training & Visualization** | 3–5   | Week 3: train the SOM with NNSOM. Week 4: Cluster Analysis. Week 5: Error Pattern Analysis                  | SOM maps, cluster purity, class–cluster intersection, error hotspots and their images  |
| **3: Novelty Detection**            | 6–8   | Week 6: novelty detection via quantization error. Week 7: U-Matrix analysis. Week 8: Comparative Evaluation | AUROC, AUPR, FPR at 95% TPR against Isolation Forest, One-Class SVM and an autoencoder |
| **4: Guided Retraining**            | 9–11  | Week 9: hard example mining. Week 10: targeted data augmentation. Week 11: iterative SOM re-analysis        | Retrained model vs. baseline; old vs. new SOM maps                                     |
| **5: Paper Writing & Code Release** | 12–14 | Final results and figures, paper draft, code release                                                        | Paper, documented code                                                                 |

Success criteria are fixed **before** the test set is examined
(to be written in `reports/`).

## Key results

**Phase 1: CNN baseline (LeNet-5 on MNIST), measured once on the held-out test split:**

| Metric              | Value                                          |
| ------------------- | ---------------------------------------------- |
| Test accuracy       | **99.18%** (10,414 / 10,500)                   |
| 95% Wilson interval | 98.99% – 99.34%                                |
| Macro F1-score      | **0.9918**                                     |
| Misclassified       | 86 test images (0.82%)                         |
| Embedding dimension | 84 (fc2 layer)                                 |
| Train / Val / Test  | 49,000 / 10,500 / 10,500 (stratified, seed 42) |

**Phase 2: SOM on the fc2 embeddings (training and validation data only; the test split is not used):**

- **Week 3, SOM training:** the official model is a 15x15 SOM, `som_15x15_nb11_ep250_s42_v2`.
  Why 15x15, and how the grid choice was checked:
  [`src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/README.md`](src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/README.md).
- **Week 4, Cluster Analysis:** every one of the 225 neurons is used, and neurons are almost pure.
  The validation purity is 0.9819, against 0.1738 when the class labels are shuffled
  (1,000 shuffles, p = 0.001).
- **Week 5, Error Pattern Analysis (error geography):** CNN errors concentrate in a few hotspot
  neurons (about 5% of the map holds 10–13 times its fair share of mistakes) and in neurons
  where digit classes mix. Both findings also hold on 10x10 and 20x20 SOMs
  ([`ERROR_PATTERN_ROBUSTNESS_RESULTS.md`](src/v1_mnist/docs/p2_som_training_visualization/error_pattern_analysis/ERROR_PATTERN_ROBUSTNESS_RESULTS.md)).

Results for Phases 3–4 (novelty detection, guided retraining) are `[TBD]` until the experiments are run.
Nothing is reported before it is measured. Changes to the original plan are logged in
[`src/v1_mnist/docs/DEVIATIONS.md`](src/v1_mnist/docs/DEVIATIONS.md).

## Reports

The results are written up as LaTeX reports. Each report folder holds its `.tex` source,
the compiled PDF and a `fig/` folder with the figures it uses.

| Phase | Report                                                                                                        | Status                                       |
| ----- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| 1     | [`cnn_baseline_report.pdf`](reports/Latex_report/v1_mnist/p1_model_training_baseline/cnn_baseline_report.pdf) | Done                                         |
| 2     | [`som_cluster_error.pdf`](reports/Latex_report/v1_mnist/p2_som_training_visualization/som_cluster_error.pdf)  | In progress; its figures are ready in `fig/` |
| 3–4   | `reports/Latex_report/v1_mnist/p3_novelty_detection/`, `p4_guided_retraining/`                                | Not started                                  |
| Final | `reports/Latex_report/v1_mnist/Final_report/`                                                                 | Not started                                  |

Technical notes: [`cnn_baseline.md`](src/v1_mnist/docs/p1_model_training_baseline/cnn_baseline_training/cnn_baseline.md)
and the other notes in [`src/v1_mnist/docs/`](src/v1_mnist/docs/).

## Repository structure

```
src/v1_mnist/
  component/                         code, by project phase and week (p = phase)
    configs/                         YAML settings (base, cnn_baseline, som, dataset/)
    pipeline/                        one runner per step, grouped by phase, + run_all.py
    p1_model_training_baseline/
      data_pipeline/                 Week 1: MNIST split, class distributions
      cnn_baseline_training/         Week 2: LeNet-5, evaluation, fc2 embeddings
    p2_som_training_visualization/
      som_nnsom_training/            Week 3: NNSOM training, quality metrics, SOM figures
      cluster_analysis/              Week 4: BMUs, purity, cluster maps
      error_pattern_analysis/        Week 5: error hotspots, robustness check
      interactive_report/            Weeks 3-5: interactive SOM HTML report
    p3_novelty_detection/            Weeks 6-8 (planned): QE novelty, U-Matrix, benchmark
    p4_guided_retraining/            Weeks 9-11 (planned): hard examples, augmentation, re-analysis
    utils/                           config, logging, run settings, shared I/O, stats, figures
  docs/                              technical notes, SOM grid-selection protocol, deviation log
  shellscripts/                      run scripts
  tests/                             pytest tests, same phase/week folders as component/
outputs/v1_mnist/    results, in the same phase/week folders (see "Data and results" below)
reports/             LaTeX reports (figures in each report's fig/ folder), Word report,
                     progress reports, proposal
demo/                interactive HTML plots (convergence plots, SOM report) and the pipeline diagram
cookbooks/           walkthrough notebooks (none yet; results are in the LaTeX reports)
presentation/        slides
research_paper/      paper drafts
```

Figures are saved as PDF only, with fonts embedded as TrueType (`pdf.fonttype` 42), by `utils/figures.py`.
No SVG, PNG or JPG files are kept in git (the one exception is the draw.io pipeline diagram).

## Installation

```bash
git clone https://github.com/nazishatta/fall-2026-group5.git
cd fall-2026-group5
python -m venv .venv
source .venv/bin/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install "dvc[s3]"
```

Then download the data, models and figures from S3 (this needs AWS access to the
`dats-capstone` bucket; run `aws configure` first):

```bash
dvc pull
```

## Data and results

MNIST downloads automatically via torchvision on first run. Fashion-MNIST and CIFAR-10 are planned.

Results in `outputs/v1_mnist/` are split between git and DVC:

- **In git:** small results only — metrics and summaries (`.json`, `.csv`), READMEs, and the
  `.dvc` pointer files.
- **In DVC** (stored on S3 at `s3://dats-capstone/shared/group5/dvc-cache`, downloaded with `dvc pull`):

| Phase | Folder (under `outputs/v1_mnist/`)                                                         | Contents                                     |
| ----- | ------------------------------------------------------------------------------------------ | -------------------------------------------- |
| 1     | `p1_model_training_baseline/cnn_baseline_training/checkpoints`                             | LeNet-5 checkpoint                           |
| 1     | `p1_model_training_baseline/cnn_baseline_training/embeddings`                              | fc2 embeddings, labels, predictions (`.npy`) |
| 1     | `p1_model_training_baseline/cnn_baseline_training/logs`                                    | training and extraction logs                 |
| 1     | `p1_model_training_baseline/cnn_baseline_training/figures`, `data_pipeline/figures`        | Phase 1 figures (PDF)                        |
| 2     | `p2_som_training_visualization/som_nnsom_training/som_models`                              | trained SOMs (10x10, 15x15, 20x20)           |
| 2     | `p2_som_training_visualization/som_nnsom_training/figures`, `convergence_plots`            | SOM figures (PDF)                            |
| 2     | `p2_som_training_visualization/cluster_analysis/figures`, `error_pattern_analysis/figures` | cluster and error figures (PDF)              |
| 2     | `p2_som_training_visualization/cluster_analysis/som_15x15_nb11_ep250_s42_v2/assignments`   | best-matching-unit assignments               |
| 2     | `p2_som_training_visualization/interactive_report`                                         | interactive SOM HTML report                  |

### Adding or updating results

When a pipeline step produces new data, models or figures, upload to S3 **before** pushing to git,
so that nobody pulls a pointer to data that is not there yet:

```bash
dvc add <changed folder>          # updates its .dvc pointer
dvc push                          # 1. upload to S3
dvc status -c                     #    should say everything is up to date
git add -A                        # 2. commit the .dvc pointers and small results
git commit -m "..."
git push
```

If a figure is used in a report, also copy it into that report's `fig/` folder and recompile the report.

## Quick start

Run from the project root, after `dvc pull`.

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
Each run has an ID, and its results record the seed, config, library versions and input hashes.
SOM run IDs encode their settings: `som_15x15_nb11_ep250_s42_v2` is a 15x15 grid,
neighborhood 11, 250 epochs, seed 42, version 2.

## Technologies

PyTorch, [NNSOM](https://amir-jafari.github.io/SOM/), scikit-learn, NumPy, Matplotlib, Plotly, pytest, DVC (S3), LaTeX

## Limitations

`[TBD - written from actual findings, not anticipated ones]`

## Team

- **Fyrooz Khan** - Project setup and repository structure, data pipeline and CNN baseline, DVC/S3 versioning, official 15x15 SOM and grid comparison, error-pattern robustness check, SOM figures and interactive report, code quality, documentation, LaTeX reports
- **Nazish Atta** - CNN evaluation and embedding extraction, initial NNSOM pipeline, cluster analysis, error-pattern and hotspot analysis, interactive convergence plots, pipeline diagram, LaTeX reports

Advisor: Dr. Amir Jafari, The George Washington University, Data Science Program.
Developed as a capstone project in the GWU MS Data Science program.
