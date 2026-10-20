# Historical SVHN Baseline Diagnostic

Recorded: 2026-10-19.

## Verified Results

| Method | MNIST mean | SVHN mean | AUROC |
|---|---:|---:|---:|
| Isolation Forest | 0.46247232 | 0.43962154 | 0.23826134 |
| Autoencoder | 0.11096986 | 0.02036880 | 0.00771872 |
| One-Class SVM | -8.99389426 | 3.84931458 | 0.96042444 |

All three AUROC values were independently reproduced from DVC-restored
historical score arrays.

## Interpretation

Isolation Forest and Autoencoder assign lower average novelty scores
to SVHN than to MNIST validation examples.

This explains their low AUROC rankings, but not the underlying cause.
Score directions must not be reversed after evaluating SVHN.

## Fairness Limitation

Historical One-Class SVM used 10,000 MNIST training embeddings.
Other fitted baselines used different training budgets.

## Leakage Control

This diagnostic only evaluated frozen historical score arrays.
No models were retrained, no thresholds were tuned, and the official
MNIST final-test set was not used.

## Provenance

Run ID: mnist_svhn_five_method_comparison_s42_initial

Score files restored through the existing score_arrays.dvc pointer.

Reproduction script:
src/v1_mnist/component/pipeline/p3_novelty_detection/inspect_svhn_scores.py
