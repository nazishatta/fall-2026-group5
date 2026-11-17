# Tests: som reanalysis (`src/v1_mnist/tests/p4_guided_retraining/som_reanalysis/`)

Tests for the code in `component/p4_guided_retraining/som_reanalysis/`. Run them with `python -m pytest src/v1_mnist/tests/p4_guided_retraining/som_reanalysis -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_analysis_and_figures.py` | The analysis stage and the figures on small synthetic cells (no PyTorch, no real run). |
| `test_msp_arm_config.py` | The secondary SOM re-analysis config (max-softmax arm) differs from the primary one only where the addendum says. |
| `test_run_state.py` | Run provenance: a first run starts, a changed input stops, an unrecorded folder is refused. |
| `test_som_cell_and_settings.py` | A tiny SOM stored and measured end to end (no PyTorch), and the real settings file. |
| `test_som_measures.py` | Map measures, CKA and the paired comparison on small hand-made examples. |
