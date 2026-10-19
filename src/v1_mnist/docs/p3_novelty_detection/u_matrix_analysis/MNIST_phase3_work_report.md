# Phase 3 novelty detection — current results and verification

Updated: 2026-10-19. MNIST validation is the in-distribution reference; Fashion-MNIST and SVHN are evaluated separately as OOD samples. These are development comparisons, not held-out final MNIST test claims.

## Scope and protocol

- Frozen LeNet-5 `fc2` embeddings (84 dimensions), selected 15×15 SOM, and training-derived QE threshold 1.5690279077294162.
- Historic SVHN five-method comparison remains immutable; supplementary MSP/energy/Mahalanobis runs are labeled separately.
- No official MNIST final-test samples were used.
- Historical details and original output references: `MNIST_phase3_historical_development_log.md`.

## Current Phase 3 verification update — 2026-10-19

This report reflects the current verified state. Earlier development-only observations and obsolete test counts are preserved separately in `MNIST_phase3_historical_development_log.md`.

### Verified Fashion-MNIST comparison

All methods use the frozen MNIST CNN checkpoint and 84-dimensional
fc2 feature representation. MNIST validation provides 10,500 ID
examples; Fashion-MNIST test provides 10,000 OOD examples.

| Method | AUROC | 95% bootstrap AUROC interval |
|---|---:|---|
| Energy | 0.995678 | [0.995057, 0.996299] |
| Max-softmax | 0.988127 | [0.986964, 0.989391] |
| kNN | 0.939197 | [0.935631, 0.942673] |
| Quantization Error | 0.918486 | [0.914368, 0.922389] |
| Mahalanobis | 0.733506 | [0.725800, 0.741031] |

The reported intervals use 2,000 stratified bootstrap iterations
with evaluation seed 2026. They quantify sampling uncertainty for
one frozen representation, not variation across independently
trained CNNs or SOMs.

All five methods were independently checked against their saved
Fashion-MNIST evaluation metrics on EC2.

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

The complete repository suite passed on EC2:
168 passed, 2 warnings, 0 failures.

Phase 3 Ruff checks passed. The earlier test counts and Phase 1
data-path errors recorded above are historical development results.

### Leakage Control

No official MNIST final-test examples were used in this Phase 3
evaluation. Models were fitted using MNIST training embeddings.
MNIST validation and external OOD sets were used for evaluation,
not post-result hyperparameter selection.

Historical fitting-budget differences, including the 10,000-sample
One-Class SVM, remain limitations of those comparisons.

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
