# Tests: utils (`src/v1_mnist/tests/utils/`)

Tests for the code in `component/utils/`. Run them with `python -m pytest src/v1_mnist/tests/utils -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_cnn_retraining.py` | The retraining trainer and the arm runner, on tiny made-up images (skipped if PyTorch is missing). |
| `test_config_run_names.py` | Configs use the proposal's names; the old week-numbered run names are refused. |
| `test_file_hash.py` | Tests for the shared file fingerprint. |
| `test_in_distribution.py` | The dataset of a Phase 3 or 4 config and the names built from it; MNIST keeps its historical names and wording. |
| `test_repo_paths.py` | Tests for the shared repository path check. |
| `test_retraining_breakdown.py` | Tests for the exploratory Phase 4 test breakdown (pure counting, no models). |
| `test_retraining_support.py` | Shared Phase 4 support code: frozen split, metrics, paired statistics, protocol stamp, arm pairs. |
| `test_run_ids.py` | Tests for the shared run-ID check. |
| `test_seed_from_config.py` | The data split seed comes only from the config (no default seed in the code). |
| `test_shared_provenance.py` | Check consistency of the frozen Phase 3 provenance manifest, and that each dataset folder uses its own manifest. |
