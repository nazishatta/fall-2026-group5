# Tests: pipeline (`src/v1_mnist/tests/pipeline/`)

Tests for the code in `component/pipeline/`. Run them with `python -m pytest src/v1_mnist/tests/pipeline -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_completion.py` | Completion records: written once after a step succeeds, never overwritten, and backfilled from a run manifest only for folders it lists. |
| `test_run_all.py` | The step runners in pipeline/ list their steps in proposal order and never overwrite results; steps a dataset's config folder lacks are skipped; the test step runs only when asked for; a study step is finished only by its final result file and a half-finished Phase 4 step continues; a Phase 3 run folder is skipped only with its completion record and stops the run without one. |
