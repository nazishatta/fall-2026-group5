# Fashion-MNIST settings

YAML settings for every Fashion-MNIST step. The config loader merges `base.yaml` from this folder into each config, so every result goes under `outputs/v1_fashion_mnist/`.

## Phases 1 and 2

Each made from the MNIST config of the same name with only the dataset-specific values changed (protocols: `docs/p1_model_training_baseline/MODEL_TRAINING_PROTOCOL.md`, `docs/p2_som_training_visualization/SOM_ANALYSIS_PROTOCOL.md`).

| Config | Step |
|---|---|
| `base.yaml` | Shared settings read by every step: dataset and model facts, paths and the seed |
| `cnn_baseline.yaml` | Phase 1: data split and LeNet-5 training |
| `som.yaml` | Phase 2: the headline 15x15 SOM, cluster and error-pattern analysis |
| `som_10x10_robustness.yaml`, `som_20x20_robustness.yaml` | Phase 2: the 10x10 and 20x20 maps for the robustness check |
| `som_optuna.yaml` | Phase 2: Optuna tuning of every grid and its 5-seed confirmation |
| `grid_comparison.yaml` | Phase 2: the pre-registered 10x10 / 15x15 / 20x20 grid comparison (MNIST's rules, unchanged) |
| `error_pattern_baselines.yaml`, `error_detector_budget.yaml` | Phase 2: the SOM's error cells against other error detectors |
| `feature_layer_ablation.yaml` | Phase 2: SOMs on fc1 and fc2, 5 SOM seeds |

## Phase 3

One config per step of the Phase 3 protocol (`docs/p3_novelty_detection/NOVELTY_DETECTION_PROTOCOL.md`), each made from the MNIST config of the same name with only the dataset-specific values changed. `phase3_shared_provenance.yaml` holds the frozen Phase 3 inputs and their SHA-256; every Phase 3 script checks its inputs against it. `novelty_statements.yaml` judges the group proposal's RQ3 and RQ4 and this protocol's statements O1 to O4 in one run. MNIST steps that only re-created or repaired MNIST's own earlier runs have no config here, so the runners skip them. Values marked `[TBD]` are filled in from earlier steps' results, in the stages described in `docs/p3_novelty_detection/README.md`.

