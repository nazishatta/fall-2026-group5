# Phase 2: SOM Training & Visualization

Trains the SOM on the frozen CNN embeddings, then studies its clusters and where the CNN's errors fall.

```
p2_som_training_visualization/
├── som_nnsom_training/            1. proposal week 3
│   ├── train_som.py               ▶ train the SOM (QE tracked every epoch)
│   ├── visualize_som.py           ▶ SOM figures: topology, U-Matrix, hits, component planes
│   ├── grid_ablation.py           ▶ 10x10 / 15x15 / 20x20 comparison table
│   ├── convergence_plots/         ▶ 3 interactive charts: QE, quality, radius (+ artifacts.py)
│   ├── trainer.py, trainer_history.py, memory_safe_init.py   training helpers
│   ├── quality_metrics.py         QE and topographic error
│   ├── run_records.py             run-ID checks, fingerprints, run manifest
│   └── som_visualizations.py, extended_plots.py              drawing helpers
├── cluster_analysis/              2. proposal week 4
│   ├── run_cluster_analysis.py    ▶ BMUs, purity, entropy, label-shuffle check
│   ├── cluster_figures.py         ▶ class maps, purity map, pie/stem
│   ├── cluster_heatmaps.py        ▶ hit, purity, entropy, dominant-class heatmaps
│   └── cluster_metrics.py         the metrics themselves
├── error_pattern_analysis/        3. proposal week 5
│   ├── error_patterns.py          ▶ error geography, hotspots, confusion pairs
│   ├── hotspot_samples.py         ▶ images of the digits in the hotspots
│   ├── error_figures.py           ▶ error-rate and complex error maps
│   ├── error_heatmaps.py          ▶ error heatmaps
│   ├── error_pattern_robustness.py          ▶ robustness statistics, one SOM at a time
│   └── error_pattern_robustness_verdict.py  ▶ applies the frozen criteria (15x15, 10x10, 20x20)
└── interactive_report/            4. after weeks 3–5
    ├── visualize_som_interactive.py   ▶ one self-contained HTML report
    ├── report_payload.py, neuron_stats.py, model_io.py, numeric_utils.py, settings.py
    ├── html_report.py             fills the page template
    └── templates/                 page, CSS, js/ (order in js/README.md)
```

**Order:** `som_nnsom_training` → `cluster_analysis` → `error_pattern_analysis` → `interactive_report`; inside each folder, top to bottom. ▶ = script you run.
Runners: `pipeline/p2_som_training_visualization/run_*.py`, or all at once with `pipeline/run_all.py`.
Results: `outputs/v1_mnist/p2_som_training_visualization/`. Each step folder except `interactive_report` also has an `*_EXPLAINED.md`.
