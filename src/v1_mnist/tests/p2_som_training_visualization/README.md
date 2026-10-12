# Phase 2: SOM Training & Visualization (Weeks 3–5) — Tests

> `p2` in the folder name means **Phase 2**.

Tests for SOM training and the cluster and error analysis of the map.

```
p2_som_training_visualization/
├── som_nnsom_training/            # Week 3: memory-safe init, trainer vs NNSOM, save path, run-ID preflight, SOM figures
│   ├── test_memory_safe_init.py
│   ├── test_trainer_vs_nnsom.py
│   ├── test_model_save_path.py
│   ├── test_run_id_preflight.py
│   └── test_som_figure_formats.py
├── cluster_analysis/              # Week 4: BMU assignment, entropy, purity, class–neuron intersection, heatmaps
│   ├── test_bmu_assignment_and_entropy.py
│   ├── test_cluster_metrics.py
│   └── test_heatmap_figure.py
└── error_pattern_analysis/        # Week 5: Wilson lower bound for neuron error rates
    └── test_error_metrics.py
```

The interactive report (Weeks 3–5) has no tests yet, so there is no `interactive_report/` folder here.
