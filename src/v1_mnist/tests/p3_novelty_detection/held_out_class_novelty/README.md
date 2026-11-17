# Tests: held out class novelty (`src/v1_mnist/tests/p3_novelty_detection/held_out_class_novelty/`)

Tests for the code in `component/p3_novelty_detection/held_out_class_novelty/`. Run them with `python -m pytest src/v1_mnist/tests/p3_novelty_detection/held_out_class_novelty -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_metrics_and_statements.py` | Per-split metrics and the statements H1 to H4 on made-up scores (NumPy and scikit-learn only). |
| `test_settings_and_split.py` | Config, class splits and image selection of the held-out-class novelty step (NumPy only). |
| `test_split_run_smoke.py` | One split end to end on tiny made-up images (skipped without PyTorch or NNSOM). |
