# Deviations and rule changes (Fashion-MNIST)

This file lists every time we did not follow a written rule for the Fashion-MNIST study, or changed one, and
what we did about it, for every phase. Oldest first. Nothing here has been hidden or undone. The rules are in
the four phase protocols:
[Phase 1](p1_model_training_baseline/MODEL_TRAINING_PROTOCOL.md) ·
[Phase 2](p2_som_training_visualization/SOM_ANALYSIS_PROTOCOL.md) ·
[Phase 3](p3_novelty_detection/NOVELTY_DETECTION_PROTOCOL.md) ·
[Phase 4](p4_guided_retraining/RETRAINING_PROTOCOL.md).

## Changes from the MNIST design, fixed before any run

These are not deviations: they were part of the protocol when it was first committed (`5dd164c`), or, for
the outside set, filled in by amendment A1 below.

| Phase | Change | Why |
|---|---|---|
| 3 | None for the outside set: SVHN test split, same file and conversion as MNIST (decided 2026-11-21, A1) | the choice was left open in the protocol's first version because MNIST itself could be an outside set for Fashion-MNIST; the professor chose SVHN, which keeps the outside set the same for both datasets |
| 4 | Macro-F1 guard re-measured | it measures the spread of the network's own results, which differs by dataset |
| 3 | MNIST's history-checking steps not repeated (Phase 3 protocol, section 3) | they re-created or audited earlier MNIST runs; Fashion-MNIST has none |
| 3 | U-Matrix and nearest-neighbour scores judged (O3, O4) in one run | the MNIST unified benchmark compared them in a historical run; here they are fixed in advance |

## Amendments and deviations

| Date | Phase | Commit | What happened | Rule broken | What we did |
|---|---|---|---|---|---|
| 2026-11-17 | All | `5dd164c` | Protocol committed as one file, `docs/FASHION_MNIST_PROTOCOL.md` (SHA-256 `dfb56d6da83b6408791ddf3944086b7370f98c511c4c21ebd4c5f338525127ca`), before any run. | Not a deviation. | First version; it stays in git history unchanged. |
| 2026-11-21 | 3 (and the Phase 2 layer ablation L3, Phase 4 novelty after retraining) | `4442611` | **A1.** Outside set fixed: SVHN test split only (Phase 3 protocol, section 2); the limitation in Phase 3 protocol section 6 and the first row of the table above updated to match. | Not a deviation: the first version left the choice to the professor, to be written down before any Phase 3 run. | The decision came from the professor, not from any result: when it was made, no Phase 3 step had run on Fashion-MNIST and no Fashion-MNIST novelty score existed. Phase 1 and 2 results were known, but they contain no outside images. With one set, "on every outside set" in O1 to O4 means "on SVHN". No run affected. |
| 2026-11-21 | 2 | `4442611` | Optuna outcome recorded in the Phase 2 protocol, section 5. | Not a deviation: no rule changed. | The Phase 2 protocol fixes the headline map as 15x15, neighbourhood 11, 250 epochs whatever Optuna finds; Optuna's setting was not better on any grid. No run affected. |
| 2026-11-21 | All | `4442611`, `2a844b2` | The one protocol file was split into one protocol per phase, placed in the phase folders as for MNIST, and the change log moved to this file. Its general parts went to `README.md` (why the study exists, where things go). | Not a deviation: no rule, threshold or setting changed; only section numbers and the cross-references between them were renumbered. | The original file is kept in git history (`git show 5dd164c:src/v1_fashion_mnist/docs/FASHION_MNIST_PROTOCOL.md`, SHA-256 above). No Fashion-MNIST run fingerprints the protocol file (the Optuna runs fingerprint the shared MNIST `OPTUNA_TUNING_PROTOCOL.md`), so no run's check is affected. No run affected. |
| 2026-11-21 | 3 | (Phase 3 configs commit) | **A2.** Before any Phase 3 run: the rules for *not met* and *inconclusive*; one step, `novelty_statements`, that judges O1 to O4 and the group proposal's RQ3 and RQ4 from all six scores in one run (the only paired QE-against-Mahalanobis comparison); `qe_random_control` and `qe_u_matrix` run as first runs; `qe_score_distribution_figure` not repeated; `qe_ood` before the benchmark (Phase 3 protocol, section 3). | Not a deviation: the protocol named the statements, the measures and the steps but not these details, and no Phase 3 result existed. | Written into section 3 before the configs were used. The shared code got the new step (MNIST runs it too, for RQ3 and RQ4 only; see the MNIST `DEVIATIONS.md`) and an earlier place for `qe_ood` in `run_all.py`, which does not change MNIST's default plan. No run affected. |
| 2026-11-21 | 3 | (Phase 3 configs commit) | Phase 3 configs written, with the frozen inputs of Phase 1 run `lenet5_fashion_mnist_s42` and the 15x15 headline SOM (SHA-256 checked against their recorded values) and Fashion-MNIST's own `phase3_shared_provenance.yaml`. Values that only an earlier Fashion-MNIST step can give (the frozen QE threshold, score-file hashes, AUROCs that later steps must reproduce, the miss count) are `[TBD]` and filled in stages after those steps run. | Not a deviation. | A script given a `[TBD]` value stops. Every filled value is copied from a saved result, never chosen. |
| 2026-11-21 | 4 | (Phase 3 configs commit) | The "Recorded outcome" placeholder of the Phase 4 protocol no longer says `[TBD]` ("Not run yet."). | Not a deviation: the protocol stamp refuses any `[TBD]`; the only one left is the macro-F1 guard, filled by its rule before the stamp. | No rule changed. |
