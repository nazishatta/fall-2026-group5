# Tests: layer ablation (`src/v1_mnist/tests/p2_som_training_visualization/som_nnsom_training/layer_ablation/`)

Tests for the code in `component/p2_som_training_visualization/som_nnsom_training/layer_ablation/`. Run them with `python -m pytest src/v1_mnist/tests/p2_som_training_visualization/som_nnsom_training/layer_ablation -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_extraction.py` | Feature-layer ablation: both layers in one pass, and the fc2 check against Phase 1 (skipped without PyTorch). |
| `test_measures_and_comparison.py` | Feature-layer ablation: measures A1 to A3 and the fc1 - fc2 verdicts on small made-up data. |
| `test_settings_and_provenance.py` | Feature-layer ablation: the real config, the run ID, the checks on layers and seeds, and the run record. |
