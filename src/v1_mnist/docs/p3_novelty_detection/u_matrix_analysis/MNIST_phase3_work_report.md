# Phase 3 novelty detection — current results and verification

Updated: 2026-10-19. MNIST validation is the in-distribution reference; SVHN is the current external OOD evaluation set. These are development comparisons, not held-out final MNIST test claims.

## Scope and protocol

- Frozen LeNet-5 `fc2` embeddings (84 dimensions), selected 15×15 SOM, and training-derived QE threshold 1.5690279077294162.
- Historic SVHN five-method comparison remains immutable; supplementary MSP/energy/Mahalanobis runs are labeled separately.
- No official MNIST final-test samples were used.
- Historical details and original output references: `MNIST_phase3_historical_development_log.md`.

## Current Phase 3 verification update — 2026-10-19

This report reflects the current verified state. Earlier development-only observations and obsolete test counts are preserved separately in `MNIST_phase3_historical_development_log.md`.

### Current evaluation scope

Following the professor's 2026-10-19 clarification, the current
comparison uses MNIST validation as in-distribution data and SVHN
as external out-of-distribution data. Exploratory Fashion-MNIST
results are deferred to later work and are not reported as current
Phase 3 findings.

### Historical SVHN baseline investigation

The saved DVC score arrays reproduce these historical AUROCs:

- Isolation Forest: 0.238261
- Feature autoencoder: 0.007719
- One-Class SVM: 0.960424

The low Isolation Forest and autoencoder AUROCs reflect their
recorded score ranking. Their underlying causes have not been
established. No score direction was changed after evaluation.

See the Phase 3 novelty-benchmark SVHN baseline diagnostic.

### FPR at 95 percent TPR

The current comparison reports the minimum observed ROC false
positive rate among operating points with TPR at least 0.95.
OOD is the positive class, and higher scores mean more novelty.

This ROC-derived evaluation operating point is different from
the frozen, training-derived quantization-error threshold.

Historical records using a different FPR definition must be
identified and reconciled before their values are combined in
a single comparison table.

### Preregistration chronology

The preregistration document states a writing date of 2026-10-11.
Its Git commit chronology does not independently establish that
the document was fixed before all earlier OOD results were viewed.

Therefore, this report does not claim prospectively registered
status for those earlier experiments without further evidence.
Historical results and protocol decisions remain unchanged.

The instructor's later validation-threshold request differs from
the original training-percentile protocol. The discrepancy is
recorded in src/v1_mnist/docs/DEVIATIONS.md.

### Current test verification

The most recently recorded full repository verification before the
current scope cleanup was 172 tests passed on EC2. This is historical
verification, not a claim that the latest commits passed the full suite.

After the MNIST-SVHN scope cleanup, 73 Phase 3 tests passed locally
with the two PyTorch-dependent test files excluded. Phase 3 Ruff
checks passed. The complete local suite is blocked during collection
because PyTorch is absent from the macOS Python environment.

### Leakage Control

No official MNIST final-test examples were used in this Phase 3
evaluation. Models were fitted using MNIST training embeddings.
MNIST validation and external OOD sets were used for evaluation,
not post-result hyperparameter selection.

Historical fitting-budget differences, including the 10,000-sample
One-Class SVM, remain limitations of those comparisons.

## Outstanding SVHN verification — 2026-10-19

The kNN novelty baseline has been integrated into a separate,
configured SVHN evaluation runner and its unit tests passed.
However, no SVHN kNN AUROC or confidence interval has yet been
measured for this new run; these values remain [TBD].

A paired stratified-bootstrap implementation for comparing QE
against max-softmax AUROC is also committed and unit-tested.
Its actual paired confidence interval remains [TBD] because
sample-aligned SVHN max-softmax score arrays are not currently
available locally. The saved individual method confidence
intervals must not be presented as a paired comparison.

Historical results, provenance, and run IDs remain unchanged.

## Additional SVHN baseline comparison — 2026-10-19

A separate, additional evaluation compared max-softmax probability
(MSP), energy, and Mahalanobis distance using the frozen MNIST CNN
and its 84-dimensional fc2 representation.

Evaluation samples:
- MNIST validation (ID): 10,500
- SVHN official test (OOD): 26,032

| Method | AUROC | AUPR | FPR@95TPR |
|---|---:|---:|---:|
| Energy | 0.999797 | 0.999917 | 0.000381 |
| Max-softmax | 0.998495 | 0.999358 | 0.007810 |
| Mahalanobis | 0.776437 | 0.785875 | 0.313143 |

FPR@95TPR is shown as a proportion rather than a percentage.

These results show that energy and MSP outperform QE on SVHN
under the evaluated configurations. The historical QE AUROC was
approximately 0.9811.

The earlier five-method benchmark remains unchanged. Its QE
ranking applies only to the methods included in that benchmark,
not to every novelty-detection method evaluated in this project.

### Reproducibility

Run ID:
mnist_svhn_required_baselines_reproduction_initial

Saved metrics:
outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/mnist_svhn_required_baselines_reproduction_initial/required_baseline_metrics.json

The runner verified frozen CNN predictions and recorded input
SHA-256 hashes, configuration hash, bootstrap settings, and
evaluation metrics.

### Limitations and leakage control

This is an additional retrospectively specified comparison,
not proof that these settings were part of the original
preregistration.

The CNN remained frozen. Mahalanobis statistics were fitted
using MNIST training embeddings only. No official MNIST
final-test samples were used, and no parameters were retuned
using these SVHN results.

This comparison uses one frozen CNN and one external OOD
dataset. It does not establish multi-seed robustness.

## Verified additional SVHN baseline results (2026-10-19)

The frozen MNIST validation (10,500 samples) versus SVHN test
(26,032 samples) benchmark was rerun with an additional kNN baseline,
k=5, using the frozen 84-dimensional CNN embeddings.

| Method | AUROC | 95% bootstrap CI |
|---|---:|---|
| QE | 0.981107 | [0.979048, 0.983089] |
| kNN | 0.976561 | [0.974306, 0.978747] |
| Max-softmax | 0.998495 | [0.998232, 0.998741] |
| Energy | 0.999797 | [0.999723, 0.999859] |
| Mahalanobis | 0.776437 | [0.768794, 0.783205] |

An aligned, stratified paired bootstrap (2,000 iterations,
seed 42) measured QE minus max-softmax AUROC as -0.017388,
with 95% confidence interval [-0.019406, -0.015418].
Max-softmax performed better on this particular comparison.

Leakage Control: The CNN and SOM were frozen. No parameters
were tuned on SVHN and the final MNIST test set was not used.
The original 95th-percentile MNIST training-QE threshold was
preserved.

Limitation: Historical QE arrays have no per-image identifiers.
Pairing assumes the identical deterministic SVHN test ordering,
preprocessing and frozen model; both independently recomputed
AUROCs match the historical values. Bootstrap uncertainty does
not capture variation across independently trained models.

Evidence:
- `mnist_svhn_required_baselines_knn_s42_v1/required_baseline_metrics.json`
- `mnist_svhn_qe_msp_paired_s42_v1/qe_msp_paired_metrics.json`

The paired result was produced using the existing
`compare_auroc_paired` implementation and frozen CNN inference.
A dedicated end-to-end paired execution runner remains to be
committed for complete reproducibility.
