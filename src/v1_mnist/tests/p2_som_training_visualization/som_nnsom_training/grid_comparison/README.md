# Tests: grid comparison (`src/v1_mnist/tests/p2_som_training_visualization/som_nnsom_training/grid_comparison/`)

Tests for the code in `component/p2_som_training_visualization/som_nnsom_training/grid_comparison/`. Run them with `python -m pytest src/v1_mnist/tests/p2_som_training_visualization/som_nnsom_training/grid_comparison -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_configs_and_stages.py` | Both datasets' configs follow the pre-registration; every stage runs end to end on toy data; nothing is overwritten; a changed input stops the run. |
| `test_endpoint.py` | The held-out endpoint and the decision rule. |
| `test_fast_trainer.py` | The fast trainer gives the same map as `train_som_with_history`; its random modes. |
