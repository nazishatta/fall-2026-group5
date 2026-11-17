# Tests: hard example mining (`src/v1_mnist/tests/p4_guided_retraining/hard_example_mining/`)

Tests for the code in `component/p4_guided_retraining/hard_example_mining/`. Run them with `python -m pytest src/v1_mnist/tests/p4_guided_retraining/hard_example_mining -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_build_pools.py` | Hard example mining: the run-ID preflight and the settings read from retraining.yaml. |
| `test_curriculum.py` | Curriculum: the schedule, the order hard images are let in, and the arms built from them. |
| `test_difficulty_and_pools.py` | Hard example mining: the three difficulty rules and the three pools, on a tiny made-up table. |
