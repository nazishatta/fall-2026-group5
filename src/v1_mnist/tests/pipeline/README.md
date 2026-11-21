# Tests: pipeline (`src/v1_mnist/tests/pipeline/`)

Tests for the code in `component/pipeline/`. Run them with `python -m pytest src/v1_mnist/tests/pipeline -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_run_all.py` | The step runners in pipeline/ list their steps in proposal order and never overwrite results; steps a dataset's config folder lacks are skipped; the test step runs only when asked for. |
