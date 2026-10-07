# Deviations from the SOM grid-selection protocol

This file lists every time we did not follow `SOM_GRID_SELECTION_PROTOCOL.md`, and what we did about it. Oldest first. Nothing here has been hidden or undone.

| Date | Commit | What happened | Rule broken | What we did |
|---|---|---|---|---|
| 2026-09-15 | a4c4b0f | The grid-selection script was written to stop with an error unless 20x20 won. | The grid must be chosen by the rule, whatever the result. | Fixed in a1b8273: that script was retired and replaced by `som_grid_ablation.py`, which only reports the numbers and picks nothing. |
| 2026-09-15 | ed8c4a8 | The 10x10, 15x15 and 20x20 runs were retrained under the **same run IDs**, replacing the files behind the 20x20 choice. On the new files the rule picks 10x10. | Results must never be overwritten. | Both result tables are recorded in the protocol's "Recorded outcome". We report that the surviving 20x20 model's fingerprint doesn't match the original. |
| 2026-09-15 | (working session) | The instructor asked us to use 15x15 as an experiment. | The chosen grid should be the one used for analysis. | Recorded in "Recorded outcome". The instructor confirmed it on issue #3 (2026-10-04). |
| 2026-09-21 | 18b5ae8 | The analysis was switched to 15x15 with no written reason, the result file `docs/week_3/SOM_GRID_SELECTION_RESULT.md` was deleted, and the model fingerprint check was removed. | The outcome must be recorded, and model files must be checked. | The result table was restored into "Recorded outcome". The fingerprint check was restored in a1b8273; it now reads the expected value from the run manifest. |
| 2026-09-27 | b908f31 | Only 15x15 was re-tuned (neighbourhood 3 to 11, epochs 100 to 250), and its metrics file was overwritten. | All grids get equal treatment; no overwriting. | We ran a fair, pre-registered test where every grid got the same tuning (abef97d, cb48b99). It chose the same setting (11) for 15x15. The old file is kept; new runs use new IDs (`_v2`). |
| 2026-10-06 | abef97d, cb48b99 | Added the pre-registered comparison (rules committed before results). | Not a deviation; this is extra analysis. | Result: inconclusive (difference +0.001, 95% range −0.100 to +0.089). See `GRID_COMPARISON_RESULTS.md`. |
