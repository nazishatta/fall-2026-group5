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

## Multi-seed fitting-budget diagnostic

The exploratory fair-budget configuration
`fair_budget_baselines.yaml` uses `seed_list` rather than `seeds`.
The runner explicitly reads `config["seed_list"]`, requires at least
five distinct seeds, and records them under `seeds` in its results.

The completed experiment used seeds 42, 43, 44, 45 and 46, with
10,000 MNIST training embeddings per method and identical selected
training indices within each seed. Isolation Forest, One-Class SVM
and autoencoder AUROC means, sample standard deviations and
per-seed values are saved in
`mnist_svhn_fair_budget_baselines_s42_46_v1/fair_budget_metrics.json`.

The experiment equalizes training-example counts, not necessarily
computation time or optimization budgets. It is exploratory and
does not replace the historical benchmark.

The original YAML field must remain unchanged because the recorded
configuration SHA-256 is part of the experiment provenance.

## Frozen configuration provenance

The canonical Phase 3 input fingerprints are maintained in
`src/v1_mnist/component/configs/phase3_shared_provenance.yaml`.
The shared verifier checks the frozen CNN checkpoint, MNIST training
and validation embeddings, and the canonical SVHN test-file checksum
for active runners that supply these inputs.

Historical experiment configurations are intentionally preserved.
Saved result JSONs include their configuration SHA-256 values, and
the original configuration files match those recorded fingerprints.

Verified preserved configurations:

- `autoencoder_standardized_diagnostic.yaml`:
  `c74e98b1bde72782fec4f921adb21965da759dcff9a00a94bff9f3f8f8ee7fec`
- `fair_budget_baselines.yaml`:
  `2d684e215472d30225b41db972783deb92cfc578f98202398278c8622990a043`
- `svhn_knn_baselines.yaml`:
  `d34d49d02f4223373177e51d9dbd1e3d7ea372796947ca861180322587fd7cc2`

Removing duplicate fingerprints directly from these original
configurations would alter their hashes and break saved-run provenance.
Future fully deduplicated configurations require distinct config
filenames and fresh run IDs. No existing run or result is overwritten.

The historical benchmark's duplicate hash fields remain an
explicit compatibility exception, not a claim that every configuration
has been fully deduplicated.
