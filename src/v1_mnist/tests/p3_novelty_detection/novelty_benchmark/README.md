# Tests: novelty benchmark (`src/v1_mnist/tests/p3_novelty_detection/novelty_benchmark/`)

Tests for the code in `component/p3_novelty_detection/novelty_benchmark/`. Run them with `python -m pytest src/v1_mnist/tests/p3_novelty_detection/novelty_benchmark -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_autoencoder_baseline.py` | Test the MNIST feature autoencoder. |
| `test_autoencoder_scaling.py` | Tests for the autoencoder scaling diagnostic (synthetic data; no torch, no test split). |
| `test_checkpoint_availability.py` | Guard Phase 3 checkpoint checks in DVC-incomplete clones. |
| `test_classical_baselines.py` | Tests for classical novelty benchmark scores. |
| `test_consolidation.py` | Validate non-destructive Phase 3 evidence consolidation. |
| `test_fair_budget_baselines.py` | Safety and protocol tests for the equal-pool baseline experiment. |
| `test_historical_baseline_verification.py` | Regression checks for historical summary comparisons and safe run creation. |
| `test_historical_svhn_inspection.py` | Tests for the configured historical SVHN inspection runner. |
| `test_knn_baseline.py` | Tests for the kNN novelty baseline. |
| `test_novelty_statements.py` | The verdict rules of statements O1 to O4 and the judged result, on small made-up scores. |
| `test_ood_comparison.py` | Small deterministic tests that never touch official MNIST test or OOD data. |
| `test_paired_bootstrap.py` | Tests for paired QE-versus-MSP bootstrap comparisons. |
| `test_provenance_validation.py` | Safety tests for frozen Phase 3 provenance. |
| `test_qe_energy_knn_paired.py` | Safety tests for the frozen QE paired-comparison runner. |
| `test_qe_norm_dependence.py` | Tests for the QE-norm dependence check. |
| `test_qe_umatrix_random_ood.py` | Tests for the Phase 3 QE/U-Matrix and random-score analysis. |
| `test_required_baselines.py` | Tests for the three required frozen-CNN novelty baselines. |
| `test_standardized_autoencoder_diagnostic.py` | Focused tests for additive Phase 3 scaler isolation. |
| `test_svhn_canonical_location.py` | Check canonical SVHN dataset location without loading datasets. |
| `test_svhn_inverted_norm_diagnostic.py` | Synthetic tests for the SVHN embedding-norm diagnostic. |
| `test_svhn_knn_config.py` | Regression checks for the separate MNIST-SVHN kNN benchmark. |
| `test_unified_novelty_benchmark.py` | Regression checks for unified nine-method evaluation protocol. |
