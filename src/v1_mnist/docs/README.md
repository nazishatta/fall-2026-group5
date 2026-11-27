# Documents (`src/v1_mnist/docs/`)

The documents follow the project proposal (`proposals/Project_proposal.md`): one folder per **phase**, and inside it one folder per **step**, with the same names as `component/`, `tests/` and `outputs/v1_mnist/`. A step folder exists here only once that step has a document.

> **Naming:** the `p` in `p1_…`, `p2_…` means **phase**.

| Phase | Folder | Documents |
|---|---|---|
| **1: Model Training Baseline** | [`p1_model_training_baseline/cnn_baseline_training/`](p1_model_training_baseline/cnn_baseline_training/README.md) | `cnn_baseline.md`: the LeNet-5 baseline, its training and results |
| **2: SOM Training & Visualization** | [`p2_som_training_visualization/som_nnsom_training/`](p2_som_training_visualization/som_nnsom_training/README.md) | Choosing the SOM grid size (protocol, pre-registered fair test, results) and the Optuna tuning protocol |
| | [`p2_som_training_visualization/error_pattern_analysis/`](p2_som_training_visualization/error_pattern_analysis/README.md) | Error-pattern robustness criteria and results, the one error-hotspot rule (`HOTSPOT_RULE.md`), and in `baselines/` the plan for the error-detector comparison |
| **3: Novelty Detection** | [`p3_novelty_detection/`](p3_novelty_detection/README.md) | QE novelty detection, U-Matrix analysis, benchmark evidence, reproducibility notes, and the held-out-digit test protocol |
| **4: Guided Retraining** | [`p4_guided_retraining/`](p4_guided_retraining/README.md) | The retraining protocol and its stamp, the SOM re-analysis protocol and results, the max-softmax-arm addendum, and the novelty-after-retraining protocol |
| All | `DEVIATIONS.md` | Every time a written rule was not followed or was changed, and what we did about it |
| All | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | One-page reproducibility statement for both datasets: environment, seeds, run IDs, DVC, how to regenerate |

## Files that are never edited

These were written **before** the results they govern. They are our evidence that the rules were not changed after the results were seen, so they stay exactly as they were, even for wording:

- `p2_som_training_visualization/som_nnsom_training/PREREGISTRATION_grid_comparison.md` and its fingerprint `prereg_stamp.txt` (changing one character would break the fingerprint)
- the rules part of `p2_som_training_visualization/som_nnsom_training/SOM_GRID_SELECTION_PROTOCOL.md` (everything before "Recorded outcome")
- `p2_som_training_visualization/error_pattern_analysis/ERROR_PATTERN_ROBUSTNESS_CRITERIA.md`
- `p4_guided_retraining/RETRAINING_PROTOCOL.md` (fingerprint in `retraining_protocol_stamp.txt`)
- the rules parts of the other protocols (`SOM_REANALYSIS_PROTOCOL.md`, `NOVELTY_AFTER_RETRAINING_PROTOCOL.md`, `HELD_OUT_CLASS_NOVELTY_PROTOCOL.md`, `OPTUNA_TUNING_PROTOCOL.md`, `MSP_ARM_ADDENDUM.md`); only their outcome sections are filled in after the run

Moving a file to another folder does not change its text, so its fingerprint still matches.

## Where things used to be

On 2026-10-10 the documents were moved from `docs/` and `docs/som/` into the phase and step folders above. Only their locations changed. Older commits, reports and notes may still mention `docs/som/...`.
