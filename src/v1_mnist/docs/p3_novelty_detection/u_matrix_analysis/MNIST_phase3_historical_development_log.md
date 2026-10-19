# Historical Phase 3 development log (archived 2026-10-19)

This preserves the original chronological work report verbatim below. Prior statements like “no OOD evaluation performed” refer to the earlier checkpoint only, not the current project status.

---

# Phase 3 MNIST novelty work report

## Scope
This report covers **MNIST-only Phase 3 work completed before presentation**.

## Completed files

### Source code
- `src/v1_mnist/component/p3_novelty_detection/qe_novelty_detection/quantization_novelty.py`
- `src/v1_mnist/component/p3_novelty_detection/qe_novelty_detection/ood_adapters.py`
- `src/v1_mnist/component/p3_novelty_detection/qe_novelty_detection/ood_metrics.py`
- `src/v1_mnist/component/p3_novelty_detection/qe_novelty_detection/ood_scoring.py`
- `src/v1_mnist/component/p3_novelty_detection/u_matrix_analysis/u_matrix_novelty.py`
- `src/v1_mnist/component/pipeline/p3_novelty_detection/run_qe_novelty.py`
- `src/v1_mnist/component/pipeline/p3_novelty_detection/run_qe_ood.py`

### Tests
- `src/v1_mnist/tests/p3_novelty_detection/qe_novelty_detection/test_quantization_novelty.py`
- `src/v1_mnist/tests/p3_novelty_detection/qe_novelty_detection/test_ood_metrics.py`
- `src/v1_mnist/tests/p3_novelty_detection/qe_novelty_detection/test_ood_runner.py`
- `src/v1_mnist/tests/p3_novelty_detection/u_matrix_analysis/test_u_matrix_novelty.py`

### Outputs
- `outputs/v1_mnist/p3_novelty_detection/qe_novelty_detection/qe_training_calibrated_initial/qe_summary.json`
- `outputs/v1_mnist/p3_novelty_detection/qe_novelty_detection/qe_random_control_evaluation_initial/qe_random_control_metrics.json`
- `outputs/v1_mnist/p3_novelty_detection/u_matrix_analysis/qe_u_matrix_validation_s42_initial/qe_u_matrix_validation_initial.json`
- `outputs/v1_mnist/p3_novelty_detection/u_matrix_analysis/qe_u_matrix_validation_initial.pdf`
- `outputs/v1_mnist/p3_novelty_detection/qe_novelty_detection/qe_training_calibrated_initial/qe_score_distribution.pdf`

## MNIST-only results

| Item | Value |
|---|---:|
| Train QE threshold percentile | 95.0 |
| Frozen QE threshold | 1.5690279077294162 |
| Training sample count | 49000 |
| Validation sample count | 10500 |
| SOM neurons | 225 |
| Validation QE mean | 0.9634832918297878 |
| Validation U-Matrix mean | 0.7809652917712436 |
| QE vs U-Matrix Spearman correlation | 0.4396258885327658 |
| Max QE reconstruction difference | 4.860474741713583e-06 |

## Leakage control
- QE threshold calibrated from **training data only**.
- U-Matrix comparison used **MNIST validation only**.
- **No MNIST test set used**.
- **No Fashion-MNIST or SVHN evaluation performed**.

## Validation
- Phase 3 tests passed locally: **25 passed**.

## Pending after presentation
- OOD dataset evaluation.
- Baseline novelty benchmarks required by the proposal.
- Final comparison tables after the team approves the next protocol step.


## MNIST benchmark development results

All three classical baselines were fitted using MNIST training embeddings
and scored on the same 10,500 MNIST validation samples.

| Method | Mean validation score |
|---|---:|
| Isolation Forest | 0.462472 |
| One-Class SVM | -8.993894 |
| Feature autoencoder | 0.110970 |

Autoencoder: 20 training epochs; final training
loss 0.112448.

These scores have different scales and cannot directly rank the methods.

### Benchmark leakage control

Models were fitted on MNIST training embeddings only. Validation data
was used for scoring, not fitting. No final MNIST test or external OOD
dataset was evaluated. AUROC, AUPR and FPR@95TPR remain [TBD].

Autoencoder functional test passed on EC2. Classical baseline test,
Ruff lint and formatting checks passed locally.

The autoencoder checkpoint was not preserved; future comparisons
require retraining under the recorded configuration and seed.


## Reproducibility verification

MNIST Phase 3 development inputs and configuration were checked using SHA-256. The frozen training and validation embedding files and benchmark configuration matched the previously recorded fingerprints. All five benchmark configuration consistency checks passed.

| Autoencoder measurement | Original | EC2 rerun |
|---|---:|---:|
| Initial training loss | 7.216251768171 | 7.216251768171 |
| Final training loss | 0.112448146944 | 0.112448146944 |
| Mean validation reconstruction error | 0.110969859425 | 0.110969859425 |
| Validation samples | 10500 | 10500 |

The repeated autoencoder run matched all reported values exactly within the tested EC2 environment. Phase 3 tests: **27 passed**; Ruff checks passed. The separate full-repository run had **118 passed, 3 skipped, 3 Phase 1 data-path permission errors**.

**Limitations:** Isolation Forest and One-Class SVM numerical reruns remain unverified. No autoencoder checkpoint or full software version manifest was saved. OOD AUROC, AUPR and FPR@95TPR remain **[TBD]**. This evidence does not certify cross-machine reproducibility or final test performance.

Machine-readable evidence: `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/mnist_validation_reproducibility_s42_initial/reproducibility_verification.json`.


## SVHN OOD evaluation and failure analysis

### Evaluation protocol and leakage control

Professor Jafari's Phase 3 novelty detection evaluation was performed
using 10,500 MNIST validation images as
in-distribution (ID) examples and 26,032 SVHN test images
as out-of-distribution (OOD) examples.

The CNN and selected 15x15 SOM remained frozen. SVHN RGB images were
converted to grayscale, resized to 28x28, and passed through the frozen
CNN fc2 feature extractor. All five methods were evaluated on the same
ID and OOD samples. Baselines were fitted only on MNIST training
embeddings, using their recorded configurations. No final MNIST test
samples were used, and no settings were retuned using SVHN results.

### Quantitative comparison

| Method | AUROC (higher better) | AUPR (higher better) | FPR@95%TPR (lower better) |
|---|---:|---:|---:|
| Quantization Error (QE) | 0.9811 | 0.9766 | 4.53% |
| U-Matrix | 0.9696 | 0.9865 | 7.01% |
| Isolation Forest | 0.2383 | 0.5552 | 87.18% |
| One-Class SVM | 0.9604 | 0.9764 | 17.94% |
| Feature autoencoder | 0.0077 | 0.4979 | 99.99% |

OOD is the positive class, with higher scores denoting greater novelty.
FPR@95%TPR is measured from the labeled evaluation ROC curve; it is
not the frozen deployment threshold.

QE achieved the highest AUROC and lowest FPR@95%TPR in this comparison.
U-Matrix achieved the highest AUPR.

The QE 95% bootstrap AUROC interval is [0.979048, 0.983089].
The experiment uses 2,000 stratified bootstrap repetitions and seed 42.

### Frozen-threshold novelty detection

The training-calibrated QE threshold was 1.569027907729.
Of 26,032 SVHN samples, 25,036
were detected and 996 were missed.
The detection rate was 96.17%.
The MNIST validation false-positive rate at the frozen threshold
was 5.13%.

### Failure-mode analysis

The 20 lowest-QE missed SVHN examples were inspected visually.
Several contain clean vertical strokes, especially digit 1, or
recognizable numeral shapes that may resemble MNIST representations.
This supports the hypothesis that their frozen CNN embeddings lie
near familiar SOM prototypes. It does not establish causality.

Missed-detection rates differed across SVHN digit classes:
digit 4, 5.71%; digit 5, 5.20%; digit 0, 0.97%.
These are descriptive findings and must not be interpreted as
proof of class-dependent causal effects.

The Isolation Forest and autoencoder showed poor original AUROCs,
0.2383 and
0.0077, respectively.
Their SVHN median novelty scores were lower than MNIST medians
under the prespecified scoring directions. This describes the
observed score inversion; it does not conclusively establish
why either method performed poorly. Their results were not
reversed, retuned, or replaced.

### Reproducibility and verification

The comparison used run ID:
`mnist_svhn_five_method_comparison_s42_initial`.

The saved comparison JSON includes the configuration, input
SHA-256 provenance, metrics, confidence intervals, and
frozen-threshold missed-example identifiers.

The complete Phase 3 automated suite passed 32 tests on EC2.
The original EC2 files were compared against the pre-evaluation
backup with no modified or deleted original files reported.

### Existing figures and results

- U-Matrix SOM-grid PDF:
  `outputs/v1_mnist/p3_novelty_detection/u_matrix_analysis/u_matrix_novelty_som_grid.pdf`
- SVHN missed-example PDF (EC2):
  `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/svhn_failure_review_s42_initial/qe_missed_svhn_examples.pdf`
- Five-method comparison metrics (EC2):
  `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/mnist_svhn_five_method_comparison_s42_initial/comparison_metrics.json`

### Limitations

This is one OOD dataset and one seed, not a general OOD robustness
claim. The One-Class SVM fitted 10,000 MNIST training examples;
the other benchmarks used 49,000. Training budgets therefore
differ. Visual review suggests possible explanations for QE
misses but does not establish their causes. Additional seeds,
fair-budget comparisons, and other OOD datasets remain future
evaluation work. No post-test retuning was performed.
