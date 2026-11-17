# Phase 2 results: SOM Training & Visualization (`outputs/v1_mnist/p2_som_training_visualization/`)

> `p2` in the folder name means **Phase 2**.

The trained SOMs and what the map shows about the CNN's classes and mistakes. The headline map is `som_15x15_nb11_ep250_s42_v2`.

```
p2_som_training_visualization/
├── som_nnsom_training/        Week 3: SOM models, logs, run_manifest.csv, QE curves, SOM figures
├── som_optuna_tuning/         Week 3: Optuna tuning of the three grid sizes and the 5-seed check
├── cluster_analysis/          Week 4: neuron assignments, purity tables, class maps
├── error_pattern_analysis/    Week 5: error geography, hotspots, robustness, error-detector comparisons
└── interactive_report/        Weeks 3–5: interactive_report.html (DVC)
```

Git keeps only the small files here (`.json`, `.csv`, `.dvc` pointers and READMEs). Larger files are downloaded with `dvc pull`; figures are made again by the pipeline.
