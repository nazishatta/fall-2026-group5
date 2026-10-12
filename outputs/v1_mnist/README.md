# MNIST v1 Outputs (`outputs/v1_mnist/`)

Results are grouped by **project phase** and, inside each phase, by **project week**, in the same order as the capstone proposal.
The folder names match `src/v1_mnist/component/` and `src/v1_mnist/tests/`, so each week's code, tests and results share one name.
Weeks 12–14 (results, paper, release) have no folder because they produce documents, not new outputs.

> **Naming:** the `p` in `p1_…`, `p2_…`, `p3_…`, `p4_…` means **phase** (e.g. `p2_som_training_visualization` = Phase 2: SOM Training & Visualization).

> **Note:** these folders were rearranged from the old `cnn_baseline/`, `som/` and `cluster/` layout (October 2026).
> The code writes to these folders; each week's runner is in `src/v1_mnist/component/pipeline/`.

## Folders in proposal order

| Order | Folder | Proposal week | What is inside |
|---|---|---|---|
| **Phase 1: Model Training Baseline** | `p1_model_training_baseline/` | | |
| 1 | `data_pipeline/` | Week 1: Environment Setup & Data Pipeline | Class-distribution charts (train/val/test), sample-image grid, split sizes (`split_stats.json`) |
| 2 | `cnn_baseline_training/` | Week 2: Train Baseline CNN | LeNet-5 checkpoint, 84-D embeddings for every split, training curves, test confusion matrix, `test_evaluation.json`, Phase 1 logs |
| **Phase 2: SOM Training & Visualization** | `p2_som_training_visualization/` | | |
| 3 | `som_nnsom_training/` | Week 3: SOM Setup with NNSOM | Trained SOM models, run logs and metrics, `run_manifest.csv`, QE histories, QE convergence curves, SOM figures (topology, hit histograms, U-Matrix, component planes, feature maps) |
| 4 | `cluster_analysis/` | Week 4: Cluster Analysis | Per-digit class maps, validation purity map, class pie and stem charts |
| 5 | `error_pattern_analysis/` | Week 5: Error Pattern Analysis | Error-rate and complex error maps, error-pattern robustness results for 10x10 / 15x15 / 20x20 and the verdict |
| — | `interactive_report/` | Weeks 3–5 together | `interactive_report.html` (map, classes, errors and samples in one page) |
| **Phase 3: Novelty Detection** | `p3_novelty_detection/` | | |
| 6 | `qe_novelty_detection/` | Week 6: Novelty Detection via Quantization Error | *(empty)* |
| 7 | `u_matrix_analysis/` | Week 7: U-Matrix & Topological Analysis | *(empty)* |
| 8 | `novelty_benchmark/` | Week 8: Comparative Evaluation | *(empty)* |
| **Phase 4: Guided Retraining** | `p4_guided_retraining/` | | |
| 9 | `hard_example_mining/` | Week 9: Hard Example Mining | *(empty)* |
| 10 | `targeted_augmentation/` | Week 10: Targeted Data Augmentation | *(empty)* |
| 11 | `som_reanalysis/` | Week 11: Iterative SOM Re-analysis | *(empty)* |

## Tree

```
outputs/v1_mnist/
├── p1_model_training_baseline/
│   ├── data_pipeline/                                   # Week 1
│   │   ├── figures/pdf/                                 # train_dist, val_dist, test_dist, sample_images
│   │   └── metrics/split_stats.json
│   └── cnn_baseline_training/                           # Week 2
│       ├── checkpoints/          (+ checkpoints.dvc)    # lenet5_best.pth
│       ├── embeddings/           (+ embeddings.dvc)     # {train,val,test}_{features,labels,preds,confidence,correct,sample_ids}.npy
│       ├── logs/                 (+ logs.dvc)           # prepare_data, train_baseline, extract_embeddings
│       ├── figures/pdf/                                 # training_curves, test_confusion_matrix
│       └── metrics/test_evaluation.json
│
├── p2_som_training_visualization/
│   ├── som_nnsom_training/                              # Week 3
│   │   ├── som_models/           (+ som_models.dvc)
│   │   ├── logs/                                        # <run_id>.log, _metrics.json, _qe_history.json, run_manifest.csv
│   │   ├── tables/                                      # <run_id>_qe_history.csv
│   │   └── figures/
│   │       ├── pdf/      convergence/                   # QE curves for every run
│   │       └── som_15x15_nb11_ep250_s42_v2/
│   │           ├── pdf/      core/                      # topology, hit histograms, U-Matrix, neuron connections
│   │           ├── pdf/      component_planes/
│   │           ├── pdf/      feature_maps/
│   │           ├── pdf/      extended/                  # feature hist/box/violin/scatter, weights
│   │           └── visualization_manifest.json
│   ├── cluster_analysis/                                # Week 4
│   │   └── figures/som_15x15_nb11_ep250_s42_v2/
│   │       ├── pdf/      class_maps/                    # digit_0 … digit_9 colour/gray hit maps
│   │       ├── pdf/      analysis_maps/                 # validation_purity_support
│   │       └── pdf/      extended/                      # class_pie, class_stem
│   ├── error_pattern_analysis/                          # Week 5
│   │   ├── figures/som_15x15_nb11_ep250_s42_v2/
│   │   │   └── pdf/      analysis_maps/                 # validation_complex_error_map, validation_error_rate_support
│   │   └── robustness/                                  # *_error_pattern_robustness.json (3 grids), error_pattern_robustness_verdict.json
│   └── interactive_report/interactive_report.html       # Weeks 3–5
│
├── p3_novelty_detection/
│   ├── qe_novelty_detection/                            # Week 6   (empty)
│   ├── u_matrix_analysis/                               # Week 7   (empty)
│   └── novelty_benchmark/                               # Week 8   (empty)
│
└── p4_guided_retraining/
    ├── hard_example_mining/                             # Week 9   (empty)
    ├── targeted_augmentation/                           # Week 10  (empty)
    └── som_reanalysis/                                  # Week 11  (empty)
```

## Things to know

- **What git keeps here:** only small metrics and records (`.json`, `.csv`), the DVC pointer files (`.dvc`), READMEs and `.gitkeep` files. Figures and the interactive report are generated by the pipeline and are not in git (removed from git on 2026-10-10; the files on disk were kept). Checkpoints, embeddings, logs and SOM models are tracked with DVC. `src/v1_mnist/tests/test_repository_rules.py` checks this.
- **Official SOM:** `som_15x15_nb11_ep250_s42_v2`. The 10x10 and 20x20 runs are only for the error-pattern robustness check. `som_15x15_seed42_v1` is a legacy run; do not use it (see `som_nnsom_training/README.md`).
- **The Week 1 log (`prepare_data.log`) is in `cnn_baseline_training/logs/`.** DVC tracks that folder as one unit, so its three logs stay together.
- **The U-Matrix figure in `som_nnsom_training/` is the Week 3 visualisation.** Week 7's U-Matrix novelty scores will go in `u_matrix_analysis/`.
- **DVC-tracked folders** (`checkpoints`, `embeddings`, `logs`, `som_models`) moved together with their `.dvc` files.
- **Not on this computer:** the headline SOM model file and the Week 4–5 analysis tables were produced on EC2, with the code from before the 2026-10-10 merge.
- `som_nnsom_training/figures/…/visualization_manifest.json` still lists the old `outputs/v1_mnist/som/...` paths; the next run of the Week 3 runner rewrites it.
- Week 4 (cluster analysis) and Week 5 (error pattern analysis) results will be written to `cluster_analysis/<model>/` (summary, `tables/`, `assignments/`, `figures/`, `metadata/`) and `error_pattern_analysis/<model>/` (summary, `tables/`, `figures/`, `representative_samples/`, `metadata/`). Each figure script writes its log to `logs/` in its week folder.
