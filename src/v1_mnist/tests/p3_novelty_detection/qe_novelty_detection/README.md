# Tests: qe novelty detection (`src/v1_mnist/tests/p3_novelty_detection/qe_novelty_detection/`)

Tests for the code in `component/p3_novelty_detection/qe_novelty_detection/`. Run them with `python -m pytest src/v1_mnist/tests/p3_novelty_detection/qe_novelty_detection -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_novelty_figures.py` | Verify Phase 3 figure conventions without writing to experiment folders. |
| `test_ood_metrics.py` | Synthetic-only checks; no actual test dataset or upstream artifact access. |
| `test_ood_runner.py` | Guardrails for the independent QE OOD evaluation runner. |
| `test_qe_random_control.py` | Deterministic tests for the Phase 3 QE random-selection control. |
| `test_qe_threshold_sweep.py` | Tests for the QE threshold sweep (synthetic data only; no test split, no torch). |
| `test_quantization_novelty.py` | Pure QE tests that do not need DVC artifacts or final test data. |
| `test_shared_config.py` | Ensure Phase 3 OOD settings inherit the shared base configuration. |
