# Tests: baselines (`src/v1_mnist/tests/p2_som_training_visualization/error_pattern_analysis/baselines/`)

Tests for the code in `component/p2_som_training_visualization/error_pattern_analysis/baselines/`. Run them with `python -m pytest src/v1_mnist/tests/p2_som_training_visualization/error_pattern_analysis/baselines -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_budget.py` | Same-budget capture of mistakes (exploratory error-detector comparison). |
| `test_detection.py` | Max-softmax, kNN-impurity and k-means scores, and that the scaler is fitted on training data only. |
| `test_hotspot_comparison.py` | Hotspots catch more errors than their share of images; bootstrap and summary helpers. |
| `test_partitions.py` | The cell error scores use training-fit rates and handle cells with no training images. |
| `test_result_tables.py` | The result table has one row per method and the summary files are written. |
| `test_settings_and_preflight.py` | Settings come from the configs, and a finished run is never overwritten. |
