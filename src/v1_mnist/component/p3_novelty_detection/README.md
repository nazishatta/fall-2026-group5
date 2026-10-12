# Phase 3: Novelty Detection

The project's core claim: the SOM's quantization error flags inputs unlike the training data (MNIST vs. Fashion-MNIST/KMNIST). No code yet.

```
p3_novelty_detection/
├── qe_novelty_detection/          1. proposal week 6: QE novelty score and threshold
├── u_matrix_analysis/             2. proposal week 7: U-Matrix view of novel samples
└── novelty_benchmark/             3. proposal week 8: compare with max-softmax, energy and
                                      Mahalanobis on the same embeddings (AUROC)
```

**Order:** `qe_novelty_detection` → `u_matrix_analysis` → `novelty_benchmark`.
Before any test data is used, fix the success criteria and thresholds in the proposal.
When code is added: a runner in `pipeline/p3_novelty_detection/`, tests in `tests/p3_novelty_detection/`, results in `outputs/v1_mnist/p3_novelty_detection/`.
