# Phase 3 Shared Pipeline Integration Handoff

## Ownership and scope

This document proposes integrating Nazish Atta's Phase 3 novelty-detection stages
into the existing shared Phase 1–2 pipeline.

No changes to `pipeline/common.py`, `pipeline/run_all.py`, or Phase 1–2
implementations are included in this handoff.

## Current implementation

The Phase 3 adapter is:

`pipeline/p3_novelty_detection/run_phase3_stage.py`

It supports:
- `--stage`
- `--config`
- `--dry-run`

Supported stages:
1. `qe_novelty` — frozen training-distance calibration and QE analysis
2. `qe_ood` — out-of-distribution evaluation
3. `svhn_benchmark` — comparison against previously frozen QE evidence

The standalone `run_ood_comparison.py` is a score-comparison utility,
not a mandatory independent pipeline stage.

## Verified evidence

- All three adapter stages generated commands successfully in dry-run mode.
- Four adapter regression tests passed.
- Five targeted novelty-comparison tests passed.
- Ruff checks passed.
- No Phase 3 benchmark was rerun during these integration changes.
- Existing experiment outputs were not overwritten.

These checks do NOT establish end-to-end pipeline execution.

## Proposed shared pipeline change — requires teammate approval

1. Register Phase 3 stages in `pipeline/common.py`, after Phase 2.
2. Assign each Phase 3 stage its own YAML configuration.
3. Have `pipeline/run_all.py` invoke the Phase 3 adapter with the correct
   `--stage`, `--config`, and optional `--dry-run` arguments.
4. Preserve the interactive report immediately after Phase 2's
   `error_pattern_analysis`, even when Phase 3 stages follow.
5. Preserve run-ID preflight checks and non-overwrite behavior.
6. Verify the complete execution plan with `--dry-run` before allowing
   execution in a compatible environment.

## Current limitations

- Phase 3 has not yet been registered in the shared pipeline.
- The historical SVHN benchmark currently uses its fixed YAML location.
- A full test-suite execution was blocked locally because the project's
  pinned PyTorch version was unavailable for the Intel macOS environment.
- Missing scientific baselines, Fashion-MNIST evaluation, and other reviewer
  requirements must be handled separately.
- Previously recorded historical evaluation metrics must not be described
  as results of a new run.

## Review request

Please review and approve the minimal changes to the two shared pipeline
registration files before integration. Do not alter existing Phase 1–2
behavior or results.
