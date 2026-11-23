# Phase 2 results: SOM Training & Visualization (`outputs/v1_mnist/p2_som_training_visualization/`)

> `p2` in the folder name means **Phase 2**.

The trained SOMs and what the map shows about the CNN's classes and mistakes. The headline map is `som_15x15_nb11_ep250_s42_v2`.

```
p2_som_training_visualization/
├── som_nnsom_training/        SOM models, logs, run_manifest.csv, QE curves, SOM figures
├── grid_comparison/           pre-registered comparison of the three grids: the original run and its reproduction
├── som_optuna_tuning/         Optuna tuning of the three grid sizes and the 5-seed check
├── feature_layer_ablation/    SOMs on fc1 and fc2, 5 SOM seeds each (DVC for the embeddings)
├── cluster_analysis/          neuron assignments, purity tables, class maps
├── error_pattern_analysis/    error geography, hotspots, robustness, error-detector comparisons
└── interactive_report/        interactive_report.html (DVC), all of Phase 2
```

Git keeps only the small files here (`.json`, `.csv`, `.dvc` pointers and READMEs). Larger files are downloaded with `dvc pull`; figures are made again by the pipeline.
