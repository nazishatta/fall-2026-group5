# Tests (`src/v1_mnist/tests/`)

The tests are grouped by **project phase**, and inside each phase by **project week**, in the same order as the capstone proposal.
Each folder holds the tests for the code written in that week.
Weeks 12–14 (results, paper, code release) have no folder because they produce no new code to test.

> **Naming:** the `p` in `p1_…`, `p2_…`, `p3_…`, `p4_…` means **phase** (e.g. `p2_som_training_visualization` = Phase 2: SOM Training & Visualization).

## Folders in proposal order

| Order | Folder | Proposal week | What the tests check |
|---|---|---|---|
| **Phase 1: Model Training Baseline** | `p1_model_training_baseline/` | | |
| 1 | `data_pipeline/` | Week 1: Environment Setup & Data Pipeline | 70/15/15 split sizes, image shape and range, all classes present, class-distribution figure saved as one PDF with embedded fonts |
| 2 | `cnn_baseline_training/` | Week 2: Train Baseline CNN | LeNet-5 output shape, `fc2` feature layer, 84-D embeddings are finite, extractor metadata, run IDs (no overwriting a CNN run) |
| **Phase 2: SOM Training & Visualization** | `p2_som_training_visualization/` | | |
| 3 | `som_nnsom_training/` | Week 3: SOM Setup with NNSOM | Memory-safe SVD init, our trainer matches NNSOM, model save path, run-ID preflight (no overwriting a run), SOM and QE-curve figures saved as one PDF each (embedded TrueType fonts, no SVG); optional dense-layer rasterizing |
| 4 | `cluster_analysis/` | Week 4: Cluster Analysis | BMU reconstruction, neuron entropy, staggered-grid coordinates, cluster purity, class–neuron intersection, heatmap saved as one PDF (no SVG) |
| 5 | `error_pattern_analysis/` | Week 5: Error Pattern Analysis | Wilson lower bound for neuron error rates |
| **Phase 3: Novelty Detection** | `p3_novelty_detection/` | | |
| 6 | `qe_novelty_detection/` | Week 6: Novelty Detection via Quantization Error | *(empty, to be added)* |
| 7 | `u_matrix_analysis/` | Week 7: U-Matrix & Topological Analysis | *(empty, to be added)* |
| 8 | `novelty_benchmark/` | Week 8: Comparative Evaluation | *(empty, to be added)* |
| **Phase 4: Guided Retraining** | `p4_guided_retraining/` | | |
| 9 | `hard_example_mining/` | Week 9: Hard Example Mining | *(empty, to be added)* |
| 10 | `targeted_augmentation/` | Week 10: Targeted Data Augmentation | *(empty, to be added)* |
| 11 | `som_reanalysis/` | Week 11: Iterative SOM Re-analysis | *(empty, to be added)* |

## Files in each folder

```
tests/
├── README.md
├── conftest.py                                      # adds the code root to the import path
├── test_repository_rules.py                         # whole-repo rules: no figures or large files in git, PDF only,
│                                                    #   no overwrite options, no seeds in code, no digit-first names
├── pipeline/                                        # the runners in component/pipeline
│   └── test_run_all.py                              # run_all.py --dry-run lists every step in order
├── utils/                                           # shared code in component/utils
│   ├── test_config_run_names.py                     # old week-numbered run names are refused
│   └── test_seed_from_config.py                     # training.seed comes only from the config
├── p1_model_training_baseline/
│   ├── data_pipeline/                               # Week 1
│   │   ├── test_data_loader.py
│   │   └── test_class_distribution_figure.py
│   └── cnn_baseline_training/                       # Week 2
│       ├── test_lenet5.py
│       ├── test_feature_extractor.py
│       ├── test_cnn_run_id.py
│       └── test_accuracy_interval.py
├── p2_som_training_visualization/
│   ├── som_nnsom_training/                          # Week 3
│   │   ├── test_memory_safe_init.py
│   │   ├── test_trainer_vs_nnsom.py
│   │   ├── test_model_save_path.py
│   │   ├── test_run_id_preflight.py
│   │   ├── test_som_figure_formats.py
│   │   ├── test_dataset_info.py
│   │   └── test_embeddings_folder.py
│   ├── cluster_analysis/                            # Week 4
│   │   ├── test_bmu_assignment_and_entropy.py
│   │   ├── test_cluster_metrics.py
│   │   ├── test_confusion_overlay.py
│   │   ├── test_heatmap_figure.py
│   │   ├── test_purity_label_shuffle.py
│   │   └── test_qe_model_check.py
│   ├── error_pattern_analysis/                      # Week 5
│   │   ├── test_error_metrics.py
│   │   ├── test_hotspot_rule.py
│   │   └── test_frozen_settings.py
│   └── interactive_report/
│       └── test_js_order.py
├── p3_novelty_detection/
│   ├── qe_novelty_detection/                        # Week 6   (empty)
│   ├── u_matrix_analysis/                           # Week 7   (empty)
│   └── novelty_benchmark/                           # Week 8   (empty)
└── p4_guided_retraining/
    ├── hard_example_mining/                         # Week 9   (empty)
    ├── targeted_augmentation/                       # Week 10  (empty)
    └── som_reanalysis/                              # Week 11  (empty)
```

Each phase folder has its own short `README.md`. Tests for shared code in `component/utils` go in `utils/`; checks that cover the whole repository stay in `test_repository_rules.py` at the top of `tests/`.

## Ideas for the empty folders

- **Week 6:** the 95th-percentile threshold is computed correctly; samples above it are flagged as novel.
- **Week 7:** the U-Matrix has one value per neuron; novelty scores rise for samples far from all prototypes.
- **Week 8:** AUROC, AUPR and FPR@95%TPR give known answers on a tiny hand-made example.
- **Week 9:** the hard-example pool contains only hotspot or boundary samples; the curriculum's hard share grows on schedule.
- **Week 10:** augmentation is applied only to samples in hotspot neurons.
- **Week 11:** the old-vs-new SOM comparison works on two tiny hand-made maps.

## How to run

Run from the code root (`Code_local/`) so imports resolve.

```bash
# all tests
python -m pytest src/v1_mnist/tests -v

# one phase
python -m pytest src/v1_mnist/tests/p2_som_training_visualization -v

# one week
python -m pytest src/v1_mnist/tests/p2_som_training_visualization/cluster_analysis -v

# one file
python -m pytest src/v1_mnist/tests/p1_model_training_baseline/cnn_baseline_training/test_lenet5.py -v
```

## Adding a test

1. Put it in the week folder (inside its phase folder) whose code it tests.
2. Name the file `test_<what_it_checks>.py`. Use a name not already used in another folder.
3. If a test checks code from two weeks, split it so each part sits in its own week's folder.
