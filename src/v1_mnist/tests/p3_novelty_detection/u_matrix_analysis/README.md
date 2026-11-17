# Tests: u matrix analysis (`src/v1_mnist/tests/p3_novelty_detection/u_matrix_analysis/`)

Tests for the code in `component/p3_novelty_detection/u_matrix_analysis/`. Run them with `python -m pytest src/v1_mnist/tests/p3_novelty_detection/u_matrix_analysis -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_figure_runners.py` | Safety tests for reproducible Phase 3 PDF figure runners. |
| `test_proposal_gap_analysis.py` | Tests for Phase 3 descriptive work; no Phase 1/2 files modified. |
| `test_qe_u_matrix.py` | Tests for Phase 3 validation-only U-Matrix reproduction. |
| `test_svhn_som_overlay.py` | Tests for the SVHN-on-SOM overlay and miss profile (no torch needed). |
| `test_u_matrix_novelty.py` | Tests for U-Matrix novelty calculations. |
