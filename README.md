# NNSOM Post-Training Analysis: MNIST and Fashion-MNIST (v1_mnist, v1_fashion_mnist)

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

The group proposal ([`proposals/Group_proposal.md`](proposals/Group_proposal.md)) turns this into three questions:

1. **Where** on the SOM do the CNN's mistakes concentrate?
2. Can the SOM **flag inputs unlike anything the model has seen** (novelty detection)?
3. Can the SOM **choose which samples to retrain on**, and does that beat choosing at random?

## Pipeline Architecture

<p align="center">
  <img src="demo/project_pipeline/nnsom_project_pipeline.drawio.svg"
       alt="NNSOM Project Pipeline Architecture"
       width="100%">
</p>

## Project phases

The phases follow the project proposal ([`proposals/Project_proposal.md`](proposals/Project_proposal.md)).

| Phase                               | What is done                                                                                                | Main outputs                                                                           |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **1: Model Training Baseline**      | Data pipeline; train a baseline CNN; extract feature embeddings for every split                             | Accuracy, precision, recall, F1; 84-D fc2 embeddings                                   |
| **2: SOM Training & Visualization** | Train the SOM with NNSOM; Cluster Analysis; Error Pattern Analysis                                          | SOM maps, cluster purity, class–cluster intersection, error hotspots and their images  |
| **3: Novelty Detection**            | Novelty detection via quantization error; U-Matrix analysis; Comparative Evaluation                         | AUROC, AUPR, FPR at 95% TPR against Isolation Forest, One-Class SVM and an autoencoder |
| **4: Guided Retraining**            | Hard example mining; targeted data augmentation; iterative SOM re-analysis                                  | Retrained model vs. baseline; old vs. new SOM maps                                     |
| **5: Paper Writing & Code Release** | Final results and figures, paper draft, code release                                                        | Paper, documented code                                                                 |

Success criteria and thresholds are fixed **before** the test set is used, in the protocol of each step
(`src/v1_mnist/docs/` and `src/v1_fashion_mnist/docs/`). Every change to a written rule is logged in
[`src/v1_mnist/docs/DEVIATIONS.md`](src/v1_mnist/docs/DEVIATIONS.md) and
[`src/v1_fashion_mnist/docs/DEVIATIONS.md`](src/v1_fashion_mnist/docs/DEVIATIONS.md).

**Datasets:** MNIST and Fashion-MNIST are both finished through Phase 4. They run through the same pipeline and
step code, each in its own versioned folders (`v1_mnist`, `v1_fashion_mnist`). SVHN is the outside (novelty)
dataset for both. The network is LeNet-5 for both; no larger network was run.

## Key results

MNIST and Fashion-MNIST side by side. Both use the same code, SOM settings and statistics; the protocols were
written before each run (`src/v1_mnist/docs/`, `src/v1_fashion_mnist/docs/`). The test split was read once for the
baseline and once in Phase 4, after the protocol was stamped.

**Phase 1: CNN baseline (LeNet-5, trained once and not tuned), measured once on the held-out test split:**

| Metric                    | MNIST                                          | Fashion-MNIST                                  |
| ------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| Test accuracy             | **99.18%** (10,414 / 10,500)                   | **90.90%** (9,544 / 10,500)                    |
| 95% Wilson interval       | 98.99% – 99.34%                                | 90.33% – 91.43%                                |
| Macro F1-score            | **0.9918**                                     | **0.909**                                      |
| Misclassified (test)      | 86 images (0.82%)                              | 956 images (9.10%)                             |
| Misclassified (validation)| 92 images                                      | 876 images                                     |
| Embedding dimension       | 84 (fc2 layer)                                 | 84 (fc2 layer)                                 |
| Train / Val / Test        | 49,000 / 10,500 / 10,500 (stratified, seed 42) | 49,000 / 10,500 / 10,500 (stratified, seed 42) |

On Fashion-MNIST the weakest class is Shirt (F1 0.746) and most mistakes are between upper-body garments.

**Phase 2: SOM on the fc2 embeddings (training and validation data only; the test split is not used):**

| Measure                                        | MNIST                                                       | Fashion-MNIST                                                |
| ---------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| Official SOM                                   | 15x15, neighbourhood 11, 250 epochs, seed 42 (`som_15x15_nb11_ep250_s42_v2`) | the same settings (`som_15x15_nb11_ep250_s42`)  |
| Weighted purity (validation)                   | 0.9819                                                      | 0.893                                                        |
| Purity with shuffled labels (1,000 shuffles)   | 0.1738 (p = 0.001)                                          | 0.173 (p = 0.001)                                            |
| Hotspots: held-out share of mistakes vs. fair share | 11.1 times (10x10: 10.5, 20x20: 13.4)                  | 3.9 times, range 3.2 to 4.4 (10x10: 3.8, 20x20: 3.8)         |
| Error detection, AUROC: SOM cell / k-means cell | 0.685 / 0.693                                              | 0.811 / 0.812                                                |
| Error detection, AUROC: kNN impurity / max-softmax | 0.874 / 0.981                                           | 0.876 / 0.906                                                |
| Grid comparison (10x10, 15x15, 20x20)          | could not separate the grids; 15x15 kept                    | 15x15 better than 20x20 at locating mistakes (−0.042)        |
| Optuna tuning                                  | the 15x15 setting is confirmed                              | no setting found is better than the 15x15 setting used       |
| Feature layer fc1 vs. fc2 (finding mistakes)   | fc2 better (error-cell AUROC 0.685 vs. 0.622)               | fc2 better (fc1 minus fc2: −0.055)                           |
| Feature layer fc1 vs. fc2 (SVHN, AUROC)        | fc1 slightly better (+0.005)                                | fc1 better (+0.087)                                          |

Why 15x15, and how the grid choice was checked:
[`src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/README.md`](src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/README.md).
Hotspots also hold on 10x10 and 20x20 SOMs
([`ERROR_PATTERN_ROBUSTNESS_RESULTS.md`](src/v1_mnist/docs/p2_som_training_visualization/error_pattern_analysis/ERROR_PATTERN_ROBUSTNESS_RESULTS.md)).
The hotspot ratios are not comparable in size: a ratio cannot exceed one divided by the overall error rate (about 114
on MNIST, about 12 on Fashion-MNIST).

**Phase 3: Novelty detection (validation images as the familiar set, SVHN test, 26,032 images, as the outside set):**

| AUROC on SVHN                         | MNIST    | Fashion-MNIST |
| ------------------------------------- | -------: | ------------: |
| Energy                                | 0.9998   | 0.982         |
| Max-softmax probability               | 0.9985   | 0.884         |
| Quantization Error (QE), the SOM      | 0.9811   | 0.789         |
| k-Nearest Neighbors (kNN)             | 0.9766   | 0.864         |
| U-Matrix, the SOM                     | 0.9696   | 0.812         |
| Mahalanobis                           | 0.7764   | 0.713         |

| Other novelty results                                   | MNIST                                              | Fashion-MNIST                                         |
| ------------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------- |
| SVHN flagged by the threshold fitted on training data   | 96.2%                                              | 3.0%                                                  |
| Near novelty (4 held-out classes, 5 splits): SOM AUROC  | mean 0.912                                         | mean 0.794                                            |
| Near novelty: SOM against max-softmax                   | inconclusive (0.912 vs. 0.944)                     | SOM better in every split (+0.046)                    |
| Near novelty: SOM against nearest-neighbour distance    | nearest neighbours better (0.966)                  | nearest neighbours better (−0.048)                    |

The MNIST AUROC values come from different historical benchmark executions and must not be read as one equal-budget
experiment; the Fashion-MNIST values all come from one run (`novelty_statements`). Both comparisons use one frozen
CNN and SOM, so they do not show multi-seed robustness (on Fashion-MNIST the SOM's AUROC on SVHN moves between 0.766
and 0.884 with the SOM seed alone). SVHN embeddings are much shorter than the normal images', which a rule on embedding
length alone already uses. The MNIST test split was not used for Phase 3. Details, baselines and leakage controls:
[`outputs/v1_mnist/p3_novelty_detection/README.md`](outputs/v1_mnist/p3_novelty_detection/README.md) and
[`outputs/v1_fashion_mnist/p3_novelty_detection/README.md`](outputs/v1_fashion_mnist/p3_novelty_detection/README.md).

**Phase 4: Guided retraining (5 seeds per arm, test split used once, after the protocol was stamped):**

| Measure                                              | MNIST                                      | Fashion-MNIST                                       |
| ---------------------------------------------------- | ------------------------------------------ | --------------------------------------------------- |
| Augmentation check (accuracy on altered images)      | first settings (shifts up to 2 px) failed; 1-px shifts used | MNIST's ranges failed (drop 0.081); halved twice, passed at drop 0.012 |
| SOM hard-image pool                                  | 4,217 images (8.6% of training set)        | 19,997 images (41%)                                 |
| Hotspot test error, standard → SOM augmentation      | 11.4% → 7.8%                               | 36.0% → 34.0%                                       |
| SOM vs. random augmentation (primary statement)      | −2.5 points: **met**                       | −1.0 point, interval includes 0: **not met**        |
| Curriculum (hard images let in slowly)               | worse (+0.45 points overall)               | worse (+3.3 points overall)                         |
| Re-analysis: hotspots shrink vs. standard training   | met                                        | not met                                             |
| Re-analysis: SOM guidance vs. random guidance        | not met                                    | met                                                 |
| Re-analysis: purer cells                             | not met                                    | not met                                             |
| SOM's AUROC on SVHN after retraining                 | 0.976 → 0.984 (no clear change)            | 0.838 → 0.715 (worse)                               |

Choosing the images the CNN is least sure about (max-softmax) worked about as well as SOM guidance on both datasets;
with 5 seeds no difference can be shown.

**In short:** on both datasets the SOM shows *where* the CNN fails and where novel inputs land, but it does not find
mistakes or far-novel images better than the CNN's own confidence or energy. Retraining on SOM-chosen images helped
on MNIST but not clearly on Fashion-MNIST, where the augmentation had to be much weaker and the SOM pool was large.

## Reports

The results are written up as LaTeX reports. Each report folder holds its `.tex` source,
the compiled PDF and a `fig/` folder with the figures it uses.

| Report | What it covers |
| ------ | -------------- |
| [`final_report.pdf` (MNIST)](reports/Latex_report/v1_mnist/Final_report/final_report.pdf) | The whole MNIST study, Phases 1 to 4 |
| [`final_report.pdf` (Fashion-MNIST)](reports/Latex_report/v1_fashion_mnist/Final_report/final_report.pdf) | The whole Fashion-MNIST study, Phases 1 to 4 |
| [`combined_report.pdf`](reports/Latex_report/v1_mnist_fashion_mnist_report/combined_report.pdf) | MNIST and Fashion-MNIST side by side |

Progress report: [`reports/Progress_Report/`](reports/Progress_Report/README.md) (named by date).
Technical notes and protocols: [`src/v1_mnist/docs/`](src/v1_mnist/docs/) and [`src/v1_fashion_mnist/docs/`](src/v1_fashion_mnist/docs/).

## Repository structure

Every folder has a README that says what is in it.

```
src/v1_mnist/
  component/                         code, by project phase and step (p = phase)
    configs/                         every setting, as YAML
    pipeline/                        one runner per step, grouped by phase, + run_all.py
    p1_model_training_baseline/
      data_pipeline/                 MNIST split, class distributions
      cnn_baseline_training/         LeNet-5, evaluation, fc2 embeddings
    p2_som_training_visualization/
      som_nnsom_training/            NNSOM training, map quality, SOM figures, Optuna tuning
      cluster_analysis/              best-matching neurons, purity, cluster maps
      error_pattern_analysis/        error hotspots, robustness, comparison with simpler detectors
      interactive_report/            interactive SOM HTML report (all of Phase 2)
    p3_novelty_detection/
      qe_novelty_detection/          quantization error as a novelty score
      u_matrix_analysis/             U-Matrix novelty score, SVHN on the map
      novelty_benchmark/             comparison with seven other detectors
      held_out_class_novelty/        harder test: 6 known digits, 4 new
    p4_guided_retraining/
      hard_example_mining/           hard-image pools, curriculum comparison
      targeted_augmentation/         retraining with altered copies of the hard images
      som_reanalysis/                new SOMs on the retrained CNNs
      novelty_after_retraining/      SVHN novelty on the retrained networks
    utils/                           config, logging, run IDs and SHA-256 checks, stats, figures, retraining
  docs/                              protocols, results, DEVIATIONS.md
  shellscripts/                      run_pipeline.sh: runs the whole pipeline (run_all.py) from any folder
  tests/                             pytest tests, same phase/step folders as component/
src/v1_fashion_mnist/
  component/                         configs/ (settings), pipeline/ (runner) and one folder per step
                                     (p1 to p4) that points to the step code shared from src/v1_mnist/
  docs/                              one protocol per phase (written before any run), results, DEVIATIONS.md
  shellscripts/                      run_pipeline.sh
  tests/                             tests for the Fashion-MNIST settings and runner
outputs/v1_mnist/    results, in the same phase/step folders (see "Data and results" below)
outputs/v1_fashion_mnist/  Fashion-MNIST results, same phase/step folders
proposals/           the project proposal and our group proposal
reports/             LaTeX final reports (MNIST, Fashion-MNIST, both datasets), progress report, Word and Markdown report folders
research_paper/      the journal paper (draft on branch feature/integrated-research-paper)
presentation/        slides
demo/                pipeline diagram and interactive HTML plots
cookbooks/           kept for walkthrough notebooks (none yet)
.github/workflows/   the automatic test run on every push and pull request
```

Figures are saved as PDF only, with fonts embedded as TrueType (`pdf.fonttype` 42), by `utils/figures.py`.
No SVG, PNG or JPG files are kept in git; the one exception is draw.io diagrams, whose `.drawio.svg` export is kept next to the `.drawio` source.

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

MNIST downloads automatically via torchvision on first run. Phase 3 uses the validation images as the familiar set and the SVHN test set (26,032 images, file `data/ood_svhn/test_32x32.mat`) as the outside set, for both datasets. Fashion-MNIST downloads the same way. A Phase 3 run folder counts as finished only when it holds its `run_complete.json`.

Results are split between git and DVC, in the same folders for both datasets:

- **In git:** small results only — metrics and summaries (`.json`, `.csv`), READMEs, and the
  `.dvc` pointer files.
- **In DVC** (stored on S3 at `s3://dats-capstone/shared/group5/dvc-cache`, downloaded with `dvc pull`), side by side:

| Phase | Contents | `outputs/v1_mnist/` | `outputs/v1_fashion_mnist/` |
| ----- | -------- | ------------------- | --------------------------- |
| 1 | LeNet-5 checkpoint | `p1_model_training_baseline/cnn_baseline_training/checkpoints` | `p1_model_training_baseline/cnn_baseline_training/checkpoints` |
| 1 | fc2 embeddings, labels, predictions (`.npy`) | `p1_model_training_baseline/cnn_baseline_training/embeddings` | `p1_model_training_baseline/cnn_baseline_training/embeddings` |
| 1 | training and extraction logs | `p1_model_training_baseline/cnn_baseline_training/logs` | `p1_model_training_baseline/cnn_baseline_training/logs` |
| 1 | Phase 1 figures (PDF) | `p1_model_training_baseline/cnn_baseline_training/figures`, `data_pipeline/figures` | `p1_model_training_baseline/cnn_baseline_training/figures`, `data_pipeline/figures` |
| 2 | trained SOMs (10x10, 15x15, 20x20) | `p2_som_training_visualization/som_nnsom_training/som_models` | `p2_som_training_visualization/som_nnsom_training/som_models` |
| 2 | SOM figures (PDF) | `p2_som_training_visualization/som_nnsom_training/figures`, `convergence_plots` | `p2_som_training_visualization/som_nnsom_training/figures`, `convergence_plots` |
| 2 | cluster and error figures (PDF) | `p2_som_training_visualization/cluster_analysis/figures`, `error_pattern_analysis/figures` | `p2_som_training_visualization/cluster_analysis/figures`, `error_pattern_analysis/figures` |
| 2 | best-matching-unit assignments | `p2_som_training_visualization/cluster_analysis/som_15x15_nb11_ep250_s42_v2/assignments` | `p2_som_training_visualization/cluster_analysis/som_15x15_nb11_ep250_s42/assignments` |
| 2 | grid comparison: maps, figure, logs | `p2_som_training_visualization/grid_comparison/grid_comparison_g10_15_20_b10_r200/` | `p2_som_training_visualization/grid_comparison/grid_comparison_g10_15_20_b10_r200/` |
| 2 | feature-layer ablation: embeddings, logs | `p2_som_training_visualization/feature_layer_ablation/<run ID>/` | `p2_som_training_visualization/feature_layer_ablation/<run ID>/` |
| 2 | interactive SOM HTML report | `p2_som_training_visualization/interactive_report` | `p2_som_training_visualization/interactive_report/interactive_report.html` |
| 3 | score arrays, figures, held-out-class networks and SOMs | `p3_novelty_detection/<step>/<run ID>/` (`*.dvc` inside) | `p3_novelty_detection/<step>/<run ID>/` (`*.dvc` inside) |
| 4 | retrained checkpoints, predictions, embeddings, new SOMs, figures | `p4_guided_retraining/<step>/<run ID>/` (`*.dvc` inside) | `p4_guided_retraining/<step>/<run ID>/` (`*.dvc` inside) |

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
# Run the test suite (both datasets)
python -m pytest src -q

# See every step without running anything
python src/v1_mnist/component/pipeline/run_all.py --from data_pipeline --dry-run

# Redraw the SOM figures from the official 15x15 SOM, then run the cluster and
# error-pattern analyses and the report (finished steps are skipped, a half-finished
# Phase 4 step continues, and results are never overwritten, so move or delete them
# to redo a step)
python src/v1_mnist/component/pipeline/run_all.py --skip-training

# The same pipeline on Fashion-MNIST (settings in src/v1_fashion_mnist/component/configs/,
# results in outputs/v1_fashion_mnist/); the shell script does the same
python src/v1_fashion_mnist/component/pipeline/run_all.py --from data_pipeline --dry-run
bash src/v1_fashion_mnist/shellscripts/run_pipeline.sh
```

The Phase 4 test step is a stop point on both datasets: the run ends at the validation summaries, then the macro-F1
guard is filled in, `run_protocol_stamp.py` is run, and only then `--to guided_retraining_test` reads the test split
(details in the pipeline README).

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

## GitHub settings (`.github/`)

[`workflows/`](.github/workflows/README.md) holds the automatic checks GitHub runs on every push and pull request.

## Technologies

PyTorch, [NNSOM](https://amir-jafari.github.io/SOM/), scikit-learn, NumPy, Matplotlib, Plotly, pytest, DVC (S3), LaTeX

## Limitations

These are the limitations found in the results (full list in the final report):

- Only 92 validation mistakes on MNIST (876 on Fashion-MNIST), so MNIST intervals are wide and AUROC differences below about 0.1 cannot be seen.
- Two datasets and one network (LeNet-5); no larger network was run, so the results may not carry over to other CNNs.
- The grid comparison could not separate 10x10, 15x15 and 20x20; 15x15 was chosen as a trade-off between purity and topographic error.
- The SVHN results use one CNN and one SOM, and the SVHN comparisons were set up after SVHN results had been seen. The held-out-digit test uses one network per split, and its splits share digits.
- Only 146 test images lie in the hotspots, and the hotspots come from the SOM, which gives the SOM-guided arms a built-in advantage on hotspot error.
- The first augmentation settings (shifts of up to 2 pixels) failed a check made before training; all reported augmentation results use 1-pixel shifts. On Fashion-MNIST the ranges had to be halved twice, which leaves the augmentation little room to help, and the SOM pool covers 41% of the training set, so the SOM and random pools overlap more than on MNIST.
- On both datasets SVHN differs from the normal data in one simple feature, the embedding length, so an AUROC on SVHN mostly measures how strongly a score follows it. The SOM's AUROC on SVHN also moves a lot with the SOM seed (0.766 to 0.884 on Fashion-MNIST).

## Team

- **Fyrooz Anika Khan** - Project setup and repository structure, data pipeline and CNN baseline, DVC/S3 versioning, official 15x15 SOM, grid comparison and Optuna tuning, error-pattern robustness check and error-detector baselines, SOM figures and interactive report, Phase 4 guided retraining, held-out-digit novelty test, code quality, documentation, LaTeX reports and final report
- **Nazish Atta** - CNN evaluation and embedding extraction, initial NNSOM pipeline, cluster analysis, error-pattern and hotspot analysis, interactive convergence plots, pipeline diagram, Phase 3 novelty detection, LaTeX reports

Advisor: Dr. Amir Jafari, The George Washington University, Data Science Program.
Developed as a capstone project in the GWU MS Data Science program.
