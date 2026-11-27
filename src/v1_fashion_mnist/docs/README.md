# Documents (`src/v1_fashion_mnist/docs/`)

The same layout as `src/v1_mnist/docs/`, following the project proposal (`proposals/Project_proposal.md`): one
folder per **phase**, and inside it one folder per **step**, with the same names as `component/`, `tests/` and
`outputs/v1_fashion_mnist/`. Each phase has one protocol, written before any Fashion-MNIST run.

> **Naming:** the `p` in `p1_…`, `p2_…` means **phase**.

| Phase | Folder | Documents |
|---|---|---|
| **1: Model Training Baseline** | [`p1_model_training_baseline/`](p1_model_training_baseline/README.md) | `MODEL_TRAINING_PROTOCOL.md`: data, split, network, embeddings, and the one test evaluation |
| **2: SOM Training & Visualization** | [`p2_som_training_visualization/`](p2_som_training_visualization/README.md) | `SOM_ANALYSIS_PROTOCOL.md`: the maps, the grid comparison, Optuna tuning (with its recorded outcome), RQ1, RQ2, the error-detector comparisons and the feature-layer ablation |
| **3: Novelty Detection** | [`p3_novelty_detection/`](p3_novelty_detection/README.md) | `NOVELTY_DETECTION_PROTOCOL.md`: the outside set (SVHN), the six scores, statements O1-O4, the further MNIST steps and the held-out classes |
| **4: Guided Retraining** | [`p4_guided_retraining/`](p4_guided_retraining/README.md) | `RETRAINING_PROTOCOL.md`: hard-image pools, arms, the macro-F1 guard, statements A1-A3 and C1-C2, the SOM re-analysis and novelty after retraining |
| All | `DEVIATIONS.md` | The changes from the MNIST design, and every amendment or deviation, with dates and commits |
| All | [`../../v1_mnist/docs/REPRODUCIBILITY.md`](../../v1_mnist/docs/REPRODUCIBILITY.md) | The reproducibility statement, shared with MNIST |

## Why this study exists

The group proposal (`proposals/Group_proposal.md`, sections 3.4 and 4) says the whole pipeline is repeated
on Fashion-MNIST. There are two reasons:

1. **More mistakes.** The MNIST network makes only 92 validation and 86 test mistakes, which makes the
   per-cell statistics weak (MNIST limitation 1 in the README).
2. **Reuse.** The professor's proposal (`proposals/Project_proposal.md`, objective 7 and deliverables)
   asks for an analysis pipeline that is "reusable across datasets/models". Running the same code on a
   second dataset, changing only the settings, is the test of that.

So the rule for every phase protocol is: **every step done for MNIST is done for Fashion-MNIST, with the MNIST
design copied unchanged, unless the dataset makes that impossible.** The results go in the same step folders
as MNIST's. Every change is listed in `DEVIATIONS.md`.

## Where things go

- Settings: `src/v1_fashion_mnist/component/configs/` (`base.yaml` sends every result to `outputs/v1_fashion_mnist/`).
- Runner: `src/v1_fashion_mnist/component/pipeline/run_all.py`. Step code: shared, in `src/v1_mnist/component/`.
- Results: `outputs/v1_fashion_mnist/<phase>/<step>/`, the same step folders as `outputs/v1_mnist/`; large files in DVC (S3), small `.json`/`.csv` in git.
- Reports: `reports/Latex_report/v1_fashion_mnist/<phase>/`.

## Files that are never edited

The rules parts of the four phase protocols (everything before "Recorded outcome") were written before the
results they govern, and stay as they are. Only their "Recorded outcome" sections, the macro-F1 guard value
(filled in by its rule) and `DEVIATIONS.md` are added to.

## Where things used to be

Until 2026-11-21 the four protocols were one file, `docs/FASHION_MNIST_PROTOCOL.md` (commit `5dd164c`). It was
split by phase with no rule changed; see `DEVIATIONS.md`. Older commits and notes may still name that file.
