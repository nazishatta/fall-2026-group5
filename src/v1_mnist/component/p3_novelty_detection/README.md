# Phase 3: Novelty Detection

Evaluates unfamiliar images using frozen MNIST CNN/SOM representations,
quantization error (QE), U-Matrix scores, and OOD baselines. MNIST
validation and SVHN are used for evaluation; the official MNIST
final-test split is excluded.

```text
p3_novelty_detection/
├── qe_novelty_detection/
│   ├── run_qe_novelty.py                 ▶ QE novelty detection
│   ├── run_qe_ood.py                     ▶ external OOD evaluation
│   ├── run_qe_random_control.py          ▶ random-selection control
│   └── quantization_novelty.py, ood_*.py, novelty_figures.py
├── u_matrix_analysis/
│   ├── run_qe_u_matrix.py                ▶ QE/U-Matrix analysis
│   ├── run_qe_umatrix_validation_figure.py ▶ validation figure
│   ├── run_u_matrix_grid_figure.py       ▶ grid visualization
│   ├── run_proposal_gap_analysis.py      ▶ proposal-gap analysis
│   └── u_matrix_novelty.py, novelty_overlay.py
└── novelty_benchmark/
    ├── run_phase3_stage.py              ▶ standalone stage launcher
    ├── run_svhn_benchmark.py             ▶ SVHN benchmark
    ├── run_ood_comparison.py             ▶ OOD comparison
    ├── run_required_baselines.py         ▶ required baselines
    ├── run_qe_energy_knn_paired.py       ▶ paired comparisons
    ├── run_qe_umatrix_random_ood.py      ▶ QE/U-Matrix control
    ├── run_standardized_autoencoder.py   ▶ autoencoder diagnostic
    ├── run_fair_budget_baselines.py      ▶ fitting-budget comparison
    ├── run_historical_baseline_verification.py ▶ baseline audit
    ├── run_missed_svhn_figure.py         ▶ missed-SVHN visualization
    └── other Python modules              scoring and analysis helpers
```

**Order:** `qe_novelty_detection` → `u_matrix_analysis` →
`novelty_benchmark`. Select the appropriate configuration for each
experiment; not every runner must execute sequentially. ▶ = runnable script.

**Runners:** `novelty_benchmark/run_phase3_stage.py` supports
`--stage`, `--config`, and `--dry-run`. The shared Phase 1/2
`pipeline/run_all.py` remains unchanged.

**Results:** `outputs/v1_mnist/p3_novelty_detection/`, separated by
step and unique run ID. Existing experiment results must not be overwritten.

See the [Phase 3 documentation](../../docs/p3_novelty_detection/README.md)
for configuration-to-run-ID traceability and scientific reports.


See the [historical provenance audit](../../docs/p3_novelty_detection/HISTORICAL_PROVENANCE_AUDIT.md)
for SHA-256 verification and unresolved historical provenance.
