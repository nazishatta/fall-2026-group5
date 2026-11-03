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
│   ├── run_qe_threshold_sweep.py         ▶ QE at several percentile cutoffs
│   └── quantization_novelty.py, qe_threshold_sweep.py, ood_*.py, novelty_figures.py
├── u_matrix_analysis/
│   ├── run_qe_u_matrix.py                ▶ QE/U-Matrix analysis
│   ├── run_qe_umatrix_validation_figure.py ▶ validation figure
│   ├── run_u_matrix_grid_figure.py       ▶ grid visualization
│   ├── run_proposal_gap_analysis.py      ▶ proposal-gap analysis
│   ├── run_svhn_som_overlay.py           ▶ SVHN on the SOM grid, where QE misses
│   └── u_matrix_novelty.py, novelty_overlay.py, svhn_som_overlay.py
└── novelty_benchmark/
    ├── run_svhn_benchmark.py             ▶ SVHN benchmark
    ├── run_ood_comparison.py             ▶ OOD comparison
    ├── run_required_baselines.py         ▶ required baselines
    ├── run_qe_energy_knn_paired.py       ▶ paired comparisons
    ├── run_qe_umatrix_random_ood.py      ▶ QE/U-Matrix control
    ├── run_standardized_autoencoder.py   ▶ autoencoder diagnostic
    ├── run_fair_budget_baselines.py      ▶ fitting-budget comparison
    ├── run_historical_baseline_verification.py ▶ baseline audit
    ├── run_missed_svhn_figure.py         ▶ missed-SVHN visualization
    ├── run_autoencoder_scaling.py        ▶ autoencoder / Isolation Forest by size and scaling
    ├── unified_novelty_benchmark.py      ▶ all nine detectors in one run
    ├── svhn_inverted_norm_diagnostic.py  ▶ SVHN embedding-norm diagnostic
    ├── inspect_svhn_scores.py            ▶ read-only SVHN score inspection
    └── other Python modules              scoring and analysis helpers
```

**Order (set in `pipeline/p3_novelty_detection/phase3_plan.py`):** `qe_novelty`, then
the whole `novelty_benchmark` runner (its `svhn_benchmark` writes the SVHN score arrays
that later steps read), then `qe_threshold_sweep`, then the whole `u_matrix_analysis`
runner. Select the appropriate configuration for each experiment. ▶ = runnable script.

**Runners:** one step runner per folder above, in
`pipeline/p3_novelty_detection/`: `run_qe_novelty_detection.py`,
`run_u_matrix_analysis.py` and `run_novelty_benchmark.py`. Each supports
`--dry-run` and `--only STEP` (with `--config` for a new run ID), and skips
steps whose results already exist. The shared `pipeline/run_all.py` runs all
three in dependency order as the step `novelty_detection`
(`run_all.py --from novelty_detection`).

**Results:** `outputs/v1_mnist/p3_novelty_detection/`, separated by
step and unique run ID. Existing experiment results must not be overwritten.

See the [Phase 3 documentation](../../docs/p3_novelty_detection/README.md)
for configuration-to-run-ID traceability and scientific reports.


See the [historical provenance audit](../../docs/p3_novelty_detection/qe_novelty_detection/HISTORICAL_PROVENANCE_AUDIT.md)
for SHA-256 verification and unresolved historical provenance.
