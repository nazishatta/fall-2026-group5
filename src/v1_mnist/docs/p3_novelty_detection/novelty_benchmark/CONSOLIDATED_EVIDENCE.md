# Phase 3 consolidated benchmark provenance

The frozen Phase 3 evaluation uses 10,500 MNIST validation samples and 26,032 SVHN test samples. This document identifies the authoritative source files so that reproduced or copied metrics do not appear to be independent experimental evidence.

**Headline MSP, energy, Mahalanobis and kNN source:** `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/mnist_svhn_required_baselines_knn_s42_v1/required_baseline_metrics.json`.

**Historical QE, U-Matrix, Isolation Forest, One-Class SVM and unscaled-autoencoder source:** `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/mnist_svhn_five_method_comparison_s42_initial/comparison_metrics.json`.

**Paired QE–MSP evidence:** `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/mnist_svhn_qe_msp_reproduction_s42_v1/qe_msp_paired_metrics.json`.

The consolidated CSV is a **descriptive joining of saved runs**, not a claim that U-Matrix and all new baselines were executed in a single identical-fit run. The historical SVM used fewer training examples than some other methods; the separate five-seed fair-budget run is exploratory and should be presented independently. No historical results are overwritten.

Historical JSON files containing `[TBD]` are **validation-only development-stage records**. Retain them unchanged for provenance. Later SVHN evaluations are recorded under distinct run IDs. Old documentation referring to 73/168/172 tests records earlier environments and commits, not the latest passing test count. Always cite each test run with the exact branch SHA, environment and command; do not replace all historical counts with a new figure.

The random-score null simulation interval is not a detector's bootstrap confidence interval. The additional standardized-autoencoder run is post-hoc exploratory and does not supersede the original unscaled autoencoder experiment. The official MNIST final-test set remains untouched until selection criteria are frozen and a guarded evaluation is implemented.

## Pipeline scope

The separate Phase 3 adapter now enumerates additional experiments. Check every `--stage` and its own `--config` with dry-run tests before claiming end-to-end integration. Shared `pipeline/common.py` and `pipeline/run_all.py` require teammate agreement; a dedicated follow-up PR was previously agreed for those shared files. Do not imply that standalone adapter coverage means shared end-to-end pipeline integration.
