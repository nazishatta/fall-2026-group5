# Addendum to the SOM re-analysis protocol: the least-confident (max-softmax) arm

Status: **written 2026-11-17, before any cell of this secondary run is computed.** It adds one arm to the
SOM re-analysis; it does not change the primary protocol (`SOM_REANALYSIS_PROTOCOL.md`), its run
(`som_reanalysis_som_15x15_nb11_ep250_s42_v2`) or its results. Commit this file and
`configs/som_reanalysis_msp_arm.yaml` before running; the run records the SHA-256 of both.

**This is a secondary analysis, added after the primary re-analysis results and the retraining test
results were seen.** It is reported as such, next to the primary statements, never in place of them.

## 1. Why

The primary re-analysis compared SOM-guided augmentation with normal training and with random
augmentation. Its result: after SOM-guided augmentation the new map's hotspots hold fewer mistakes than
after normal training (R2 met), but not clearly fewer than after random augmentation (R4 not met).
The retraining test also found that augmenting the **least-confident** images (lowest max-softmax)
lowered the hotspot error about as much as SOM guidance did. The open question:

1. **Does confidence-guided augmentation also shrink the new map's weak regions?** If yes, the smaller
   hotspots of the new map are an effect of augmenting hard images in general, not of SOM guidance.
2. **Is SOM guidance better than confidence guidance on the new map?**

## 2. What changes (everything else is the primary protocol, sections 2 to 8)

- **Arms:** `standard`, `augmented_som`, `augmented_msp` of the targeted augmentation run
  (`configs/retraining.yaml`, section `augmentation`), seeds 1 to 5. `augmented_msp` adds the same number
  of altered copies (12,651) as `augmented_som`, chosen by lowest max-softmax confidence.
- **Pairs:** (`augmented_som`, `standard`), (`augmented_som`, `augmented_msp`), (`augmented_msp`, `standard`).
- **Run ID:** `som_reanalysis_msp_arm_som_15x15_nb11_ep250_s42_v2` (config `run.name`); same code, SOM
  recipe, measures M1 to M8, hotspot rule, paired t-interval and 1.0-point M4 tolerance as the primary run.
- The `standard` and `augmented_som` cells are re-computed (the code needs every compared arm in one run).

## 3. Statements (the primary rules, applied to the new pairs)

| Statement | Pair | Rule (primary protocol section 5) |
|---|---|---|
| S1: confidence guidance also shrinks the hotspots | `augmented_msp` − `standard` | M5 difference below 0 and its interval excludes 0 (the R2 rule) |
| S2 (= R4 of this run): SOM guidance shrinks the hotspots more than confidence guidance | `augmented_som` − `augmented_msp` | as R2 |
| S3 (= R3 of this run): SOM guidance sharpens the boundaries more than confidence guidance | `augmented_som` − `augmented_msp` | M1 difference above 0 and its interval excludes 0 (the R1 rule) |

R1, R2 and R5 of this run repeat the primary pair (`augmented_som` − `standard`); they are a
**reproduction check**, not new results (section 4). In `tables/statements.json` of this run the names
`R3_..._vs_random` and `R4_..._vs_random` refer to `augmented_msp` (the second pair), not to random
augmentation. S1 is read from `tables/paired_comparisons.json` (pair `augmented_msp_vs_standard`).

How the outcomes are read (fixed now):

- S1 met and S2 not met: smaller weak regions on the new map come from augmenting hard images, whichever
  way they are chosen; SOM guidance adds nothing measurable here.
- S1 not met and S2 met: SOM guidance shrinks the map's weak regions where confidence guidance does not.
- Neither met: with 5 seeds, no difference can be shown.

## 4. Reproduction check (reported, not judged)

Same checkpoints, same SOM recipe and seed: the `standard` and `augmented_som` cells should reproduce the
primary run's measures. Reported: the largest absolute difference of M1 to M7 between the two runs, per
cell. Small differences can come from GPU arithmetic during the embedding extraction.

## 5. Limitations

As the primary protocol (one SOM seed per cell, 5 CNN seeds, only large effects detectable), plus:
the arm was added after the results were seen, and the hard-image pool of `augmented_som` came from the
Phase 2 SOM, so the SOM arm may share structure with the map's hotspots that the max-softmax arm does not.

## 6. Leakage control

As the primary protocol: training and validation images only; the scaler and SOM are fitted on training
embeddings; nothing is tuned; the MNIST test split is not read.

## 7. Recorded outcome

[TBD: filled in after the run, from `tables/`; sections 1 to 6 stay unchanged.]
