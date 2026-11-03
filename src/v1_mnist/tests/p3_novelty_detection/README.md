# Phase 3: Novelty Detection Tests

The Phase 3 test suite validates novelty scoring, U-Matrix analysis,
OOD evaluation, benchmark implementations, reproducibility safeguards,
and integration with the shared project pipeline.

## Test organization

```text
p3_novelty_detection/
├── qe_novelty_detection/
│   ├── test_novelty_figures.py
│   ├── test_ood_metrics.py
│   ├── test_ood_runner.py
│   ├── test_qe_random_control.py
│   ├── test_quantization_novelty.py
│   └── test_shared_config.py
├── u_matrix_analysis/
│   ├── test_figure_runners.py
│   ├── test_proposal_gap_analysis.py
│   ├── test_qe_u_matrix.py
│   └── test_u_matrix_novelty.py
└── novelty_benchmark/
    ├── test_autoencoder_baseline.py
    ├── test_checkpoint_availability.py
    ├── test_classical_baselines.py
    ├── test_cli.py
    ├── test_consolidation.py
    ├── test_fair_budget_baselines.py
    ├── test_historical_baseline_verification.py
    ├── test_historical_svhn_inspection.py
    ├── test_knn_baseline.py
    ├── test_novelty_pipeline.py
    ├── test_ood_comparison.py
    ├── test_ood_comparison_cli.py
    ├── test_paired_bootstrap.py
    ├── test_provenance_validation.py
    ├── test_qe_energy_knn_paired.py
    ├── test_qe_umatrix_random_ood.py
    ├── test_required_baselines.py
    ├── test_stage_dispatch.py
    ├── test_standardized_autoencoder_diagnostic.py
    ├── test_svhn_canonical_location.py
    ├── test_svhn_inverted_norm_diagnostic.py
    ├── test_svhn_knn_config.py
    └── test_unified_novelty_benchmark.py
```

Shared provenance tests are maintained separately under
`tests/utils/test_shared_provenance.py`.

The original Phase 1/2 pipeline tests remain under
`tests/pipeline/`.

## Test responsibilities

- qe_novelty_detection: Quantization-error scoring, OOD metrics,
  random controls, runner behavior, and configuration validation.
- u_matrix_analysis: U-Matrix calculations, novelty analysis,
  proposal-gap analysis, and figure runners.
- novelty_benchmark: Baseline implementations, SVHN evaluation,
  benchmark consolidation, paired comparisons, provenance checks,
  and shared-pipeline integration.

## Verified EC2 checks

Recorded: 2026-11-01.

- Full repository suite: 260 passed, 2 warnings, 0 failures.
- Phase 3 scientific tests: 125 passed in the separately executed
  scientific test suite before pipeline-test relocation.
- Phase 3 pipeline integration tests: 36 passed.
- Phase 1/2 pipeline regression tests: 3 passed.
- Full repository suite passed again after test relocation.
- Staged and unstaged Git whitespace checks passed.

The two warnings originate from existing Seaborn and NNSOM code
in Phase 1 and Phase 2. Neither caused a test failure.

## Environment and reproducibility limitations

The local Intel macOS environment previously could not collect
the complete suite using the repository's pinned PyTorch version.
The full suite was subsequently verified on the compatible EC2
environment.

Passing software tests establishes code-test success in the
verified environment. It does not establish cross-platform
numerical equivalence, multiple independent model-training seeds,
or resolution of every historical baseline concern.

## Benchmark provenance clarification

The consolidated nine-method CSV contains results from two source
runs, not a newly fitted single execution. It preserves each method's
`source_run_id`. QE and U-Matrix were evaluated together in the original
five-method benchmark; max-softmax, energy, Mahalanobis and kNN were
evaluated in the subsequent frozen-CNN baseline run.

The equal-fitting-budget diagnostic uses five explicit seeds, stored
under `seed_list` in its original configuration. That original
configuration is retained because its SHA-256 is recorded in the
committed experiment metrics. This is a naming compatibility
exception to the requested `seeds` key, not evidence that the
experiment was executed without a seed list.
