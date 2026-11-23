# `p2_som_training_visualization` results (Fashion-MNIST)

Phase 2, SOM training and visualization: the SOM on the Fashion-MNIST embeddings and what it shows about the
network's classes and mistakes. The headline map is `som_15x15_nb11_ep250_s42` (MNIST's settings, unchanged).

```
p2_som_training_visualization/
├── som_nnsom_training/        the 10x10, 15x15 and 20x20 maps (DVC), logs, QE curves, grid table, SOM figures
├── grid_comparison/           pre-registered comparison of the three grids (maps, figure and logs with DVC)
├── som_optuna_tuning/         Optuna tuning of the three grid sizes and the 5-seed check
├── feature_layer_ablation/    SOMs on fc1 and fc2, 5 SOM seeds each
├── cluster_analysis/          purity against the label-shuffled baseline, class maps (RQ1)
├── error_pattern_analysis/    where the mistakes sit: hotspots, robustness on 10x10 and 20x20, error detectors (RQ2)
└── interactive_report/        interactive_report.html (DVC), all of Phase 2
```

Protocol and recorded results: `src/v1_fashion_mnist/docs/p2_som_training_visualization/SOM_ANALYSIS_PROTOCOL.md`
(section 5). Each folder's README names the code and config that wrote it.

Git keeps only the small files here (`.json`, `.csv`, `.dvc` pointers and READMEs). Maps, figures and logs are
kept with DVC: run `dvc pull` to get them. Results are never overwritten: a repeated step gets a new run ID.
