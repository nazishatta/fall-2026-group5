# Documents (`src/v1_mnist/docs/`)

The documents follow the project proposal (`reports/Project_proposal.md`): one folder per **phase**, and inside it one folder per **week**, the same as `component/`, `tests/` and `outputs/v1_mnist/`.

> **Naming:** the `p` in `p1_…`, `p2_…` means **phase**. A week folder exists here only once that week has a document.

| Phase | Week | Folder | Documents |
|---|---|---|---|
| **1: Model Training Baseline** | 2 | `p1_model_training_baseline/cnn_baseline_training/` | `cnn_baseline.md`: LeNet-5 baseline, training and results |
| **2: SOM Training & Visualization** | 3 | `p2_som_training_visualization/som_nnsom_training/` | Choosing the SOM grid size: protocol, pre-registered fair test, results ([README](p2_som_training_visualization/som_nnsom_training/README.md)) |
| | 5 | `p2_som_training_visualization/error_pattern_analysis/` | Error-pattern robustness check: criteria and results; the one error-hotspot rule (`HOTSPOT_RULE.md`) ([README](p2_som_training_visualization/error_pattern_analysis/README.md)) |
| All | – | `DEVIATIONS.md` | Every time a written rule was not followed, or was changed, and what we did about it |

## Files that are never edited

These were written **before** the results they govern. They are our evidence that we did not change the rules after seeing the results, so they stay exactly as they were, even for wording:

- `p2_som_training_visualization/som_nnsom_training/PREREGISTRATION_grid_comparison.md` and its fingerprint `prereg_stamp.txt` (changing one character would break the fingerprint)
- the rules part of `p2_som_training_visualization/som_nnsom_training/SOM_GRID_SELECTION_PROTOCOL.md` (everything before "Recorded outcome")
- `p2_som_training_visualization/error_pattern_analysis/ERROR_PATTERN_ROBUSTNESS_CRITERIA.md`

Moving them to another folder does not change their text, so their fingerprints still match.

## Where things used to be

On 2026-10-10 the documents were moved from `docs/` and `docs/som/` into the phase and week folders above. Only their locations changed (`DEVIATIONS.md` also gained a "Week" column). Older commits, reports and notes may still mention `docs/som/...`.
