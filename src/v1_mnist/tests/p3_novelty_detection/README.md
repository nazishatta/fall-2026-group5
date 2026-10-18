# Phase 3: Novelty Detection Tests

The Phase 3 test directory is no longer empty.

Test coverage includes novelty-score calculations, OOD comparison helpers,
and the standalone Phase 3 pipeline adapter.

## Locally verified checks

- `test_phase3_adapter.py`: four tests passed.
- `novelty_benchmark/test_ood_comparison.py`: five tests passed.
- Selected Phase 3 Ruff checks passed.

These results are targeted checks, not a full-suite pass.

## Environment limitation

The full Phase 3 suite could not be collected successfully on the
local Intel macOS Python 3.11 environment because the repository-pinned
PyTorch version was unavailable for that platform.

The complete test suite still requires verification in a compatible
environment before claiming full reproducibility.
