# Code (`src/v1_mnist/component/`)

The code is arranged by **project phase** and, inside each phase, by **step**, in the order of the project proposal (`reports/Project_proposal.md`). `tests/`, `docs/` and `outputs/v1_mnist/` use the same folder names, so each step's code, tests, documents and results line up.

> **Naming:** the `p` in `p1_…`, `p2_…`, `p3_…`, `p4_…` means **phase**.

## How to run

One runner per step lives in `pipeline/`; see [`pipeline/README.md`](pipeline/README.md). From the code root (the folder that holds `src/` and `outputs/`):

```bash
python src/v1_mnist/component/pipeline/run_all.py --from data_pipeline --dry-run   # list every step, run nothing
python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_cluster_analysis.py
```

## Folders in proposal order

| Phase | Proposal week | Folder | What the code does |
|---|---|---|---|
| **1: Model Training Baseline** | 1 | [`p1_model_training_baseline/data_pipeline/`](p1_model_training_baseline/data_pipeline/README.md) | Load MNIST, stratified 70/15/15 split, class-distribution charts, sample images |
| | 2 | [`p1_model_training_baseline/cnn_baseline_training/`](p1_model_training_baseline/cnn_baseline_training/README.md) | LeNet-5, training, test evaluation, 84-number fc2 embeddings |
| **2: SOM Training & Visualization** | 3 | [`p2_som_training_visualization/som_nnsom_training/`](p2_som_training_visualization/som_nnsom_training/README.md) | NNSOM training with QE history, map quality, SOM figures, grid-size comparison, Optuna tuning |
| | 4 | [`p2_som_training_visualization/cluster_analysis/`](p2_som_training_visualization/cluster_analysis/README.md) | Best-matching neurons, purity, entropy, class-neuron intersection, label-shuffle control |
| | 5 | [`p2_som_training_visualization/error_pattern_analysis/`](p2_som_training_visualization/error_pattern_analysis/README.md) | Where the mistakes fall, hotspots, robustness on three grids, comparison with simpler error detectors |
| | 3–5 | [`p2_som_training_visualization/interactive_report/`](p2_som_training_visualization/interactive_report/README.md) | One interactive HTML report of the SOM |
| **3: Novelty Detection** | 6 | [`p3_novelty_detection/qe_novelty_detection/`](p3_novelty_detection/qe_novelty_detection/README.md) | Quantization error as a novelty score, MNIST against SVHN |
| | 7 | [`p3_novelty_detection/u_matrix_analysis/`](p3_novelty_detection/u_matrix_analysis/README.md) | U-Matrix novelty score, where SVHN lands on the map |
| | 8 | [`p3_novelty_detection/novelty_benchmark/`](p3_novelty_detection/novelty_benchmark/README.md) | QE and U-Matrix against seven other detectors, with paired intervals |
| | – | [`p3_novelty_detection/held_out_class_novelty/`](p3_novelty_detection/held_out_class_novelty/README.md) | Harder test: train on 6 digits, detect the other 4 |
| **4: Guided Retraining** | 9 | [`p4_guided_retraining/hard_example_mining/`](p4_guided_retraining/hard_example_mining/README.md) | Which training images are hard; equal-size SOM, max-softmax and random pools; curriculum comparison |
| | 10 | [`p4_guided_retraining/targeted_augmentation/`](p4_guided_retraining/targeted_augmentation/README.md) | Altered copies of the hard images, retraining arms, hotspot and elsewhere error |
| | 11 | [`p4_guided_retraining/som_reanalysis/`](p4_guided_retraining/som_reanalysis/README.md) | New SOMs on the retrained CNNs, old map against new map |
| | – | [`p4_guided_retraining/novelty_after_retraining/`](p4_guided_retraining/novelty_after_retraining/README.md) | Does the SOM still detect SVHN after retraining? |

Shared by every step:

| Folder | What is inside |
|---|---|
| [`pipeline/`](pipeline/README.md) | One runner per step (`p1_…/` to `p4_…/`), `run_all.py`, `common.py` |
| [`configs/`](configs/README.md) | Every setting, as YAML (one line per file in its README) |
| [`utils/`](utils/README.md) | Shared helpers: config loading, logging, run IDs and SHA-256 checks, SOM loading, statistics, figures, the hotspot rule, the Phase 4 retraining code |

Dataset and model facts (number of classes, embedding size, layer name, names used in figure titles and file names) are set once in `configs/base.yaml` under `dataset:` and `model:`. The Phase 2 code reads them through `utils/dataset_info.py` instead of hard-coding MNIST values. Phase 1 and `hotspot_samples.py` stay MNIST-specific on purpose.

## Adding code for a new step

1. Put the code in that step's folder (for example `p3_novelty_detection/qe_novelty_detection/`) and its settings in `configs/`.
2. Write its outputs to the matching folder under `outputs/v1_mnist/`, under a new run ID.
3. Add its tests to the matching folder under `tests/`.
4. Add a runner in `pipeline/p<phase>_…/`, named after the step folder, and register it in `pipeline/common.py` (`STEP_RUNNERS`, in proposal order) so `run_all.py` runs it, or list it as a standalone runner in `pipeline/README.md`.
5. Add or update the README of the folder.
