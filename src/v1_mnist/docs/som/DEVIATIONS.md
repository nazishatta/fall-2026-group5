# Deviations from the SOM grid-selection protocol

Every departure from `SOM_GRID_SELECTION_PROTOCOL.md`, with what was done about it. Newest last. Nothing listed here has been hidden or reverted.

| Date | Commit | What happened | Rule affected | What we did |
|---|---|---|---|---|
| 2026-09-15 | a4c4b0f | Selection script written to raise an error unless 20x20 wins | Selection must follow the frozen rule, whatever the outcome | Hard-coded check to be removed; the ablation script now reports whichever grid wins |
| 2026-09-15 | ed8c4a8 | 10x10, 15x15 and 20x20 candidate runs retrained under the **same run IDs**, replacing the artifacts behind the 20x20 selection; committed selection table now shows 10x10 | Artifacts must never be silently overwritten | Both tables recorded in the Selection Outcome; surviving 20x20 model hash mismatch reported |
| 2026-09-15 | (working session) | Instructor asked for 15x15 as an experiment | Selected grid should be used for analysis | Recorded in Selection Outcome; confirmed by instructor on issue #3 (2026-10-04) |
| 2026-09-21 | 18b5ae8 | Analysis switched to 15x15 without a written reason; `docs/week_3/SOM_GRID_SELECTION_RESULT.md` deleted; `EXPECTED_MODEL_SHA256` check removed | Selection outcome must be recorded; artifact integrity | Result table restored into Selection Outcome; hash check to be restored, reading the expected hash from a run manifest |
| 2026-09-27 | b908f31 | 15x15 alone re-tuned (`init_neighborhood` 3 to 11, `epochs` 100 to 250); `som_15x15_seed42_final_metrics.json` overwritten in place | Equal treatment of candidates; no overwrites | Equal-budget comparison pre-registered (abef97d) and run (cb48b99); same setting (11) confirmed for 15x15 under a budget all grids shared; legacy file kept, new runs use new IDs |
| 2026-10-06 | abef97d, cb48b99 | Pre-registered held-out comparison added (rules committed before results) | Not a deviation; added analysis | Result: inconclusive (AUROC difference +0.001, 95% interval -0.100 to +0.089) |
