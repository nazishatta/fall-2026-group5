# Unified Nine-Method Novelty Benchmark

Run ID: `mnist_svhn_unified_nine_methods_s42_46_v1`

## Evaluation

MNIST validation: 10,500 samples; SVHN official test: 26,032 samples.
This is a post-hoc exploratory comparison, not a final locked-test result.

## AUROC results

| Method | AUROC | Evaluation |
|---|---:|---|
| qe | 0.981107 | Frozen reference |
| u_matrix | 0.969629 | Frozen reference |
| max_softmax | 0.998495 | Frozen reference |
| energy | 0.999797 | Frozen reference |
| isolation_forest | 0.280102 ± 0.084964 | Five-seed mean ± SD |
| one_class_svm | 0.961593 ± 0.002353 | Five-seed mean ± SD |
| autoencoder | 0.979042 ± 0.017246 | Five-seed mean ± SD |
| mahalanobis | 0.807632 ± 0.011178 | Five-seed mean ± SD |
| knn | 0.960716 ± 0.008020 | Five-seed mean ± SD |

## Leakage Control

All freshly fitted methods use MNIST training subsets only; scalers fit on training only. Validation and SVHN are evaluation-only. SVHN previously examined, therefore post-hoc exploratory. MNIST final test never accessed.

## Limitations

- Frozen CNN and SOM were trained on historical complete training data, not retrained for each 10k subset.
- Five fitted methods share exact selected MNIST training rows, but not computational budgets or model families.
- Isolation Forest uses 256 samples per tree; total training pool has 10k rows.
- Bootstrap intervals of frozen methods condition on one frozen CNN/SOM; seed spread of fitted methods is not a bootstrap confidence interval.
- All methods use SVHN as external OOD; this evaluation was informed by prior SVHN observations.
- Frozen SOM preprocessing and model selection follow historical settings; no new selection or tuning performed.

## Reproducibility

Configuration: `src/v1_mnist/component/configs/unified_novelty_benchmark.yaml`.

Detailed metrics, run provenance, training-subset fingerprints, and uncertainty are recorded in `unified_nine_method_metrics.json`.

The existing frozen CNN/SOM and historical experiment outputs were not modified.
