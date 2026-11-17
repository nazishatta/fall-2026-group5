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

Run `som_reanalysis_msp_arm_som_15x15_nb11_ep250_s42_v2`, finished 2026-11-17 on EC2. Source:
`tables/paired_comparisons.csv`, `tables/statements.json` and `tables/cell_measures.csv` of that run.
Differences are first arm minus second arm, mean over 5 seeds, with the 95% paired t-interval.

| Statement | Measure | Difference | 95% interval | Seeds with first arm lower | Outcome |
|---|---|---|---|---|---|
| S1: `augmented_msp` − `standard` | M5 hotspot error rate | −0.036 | −0.093 to +0.021 | 3 of 5 | **not met** (interval includes 0) |
| S2: `augmented_som` − `augmented_msp` | M5 hotspot error rate | −0.018 | −0.076 to +0.039 | 3 of 5 | **not met** |
| S3: `augmented_som` − `augmented_msp` | M1 weighted purity | −0.0006 | −0.0029 to +0.0017 | (higher in 2 of 5) | **not met** |

Mean hotspot error rate of the new maps: `standard` 0.229, `augmented_msp` 0.193, `augmented_som` 0.175.
Mean weighted purity: 0.982, 0.983 and 0.982.

**Reading (section 3 rule "neither met"):** with 5 seeds, no difference can be shown between SOM guidance
and confidence guidance on the new map. Only SOM-guided augmentation has a clear drop against normal
training (R2, below), but it is not clearly lower than confidence-guided augmentation, so the result does
not show that SOM guidance shrinks the map's weak regions more than augmenting the least-confident images.

**Reproduction check (section 4):** the `standard`, `augmented_som` and `baseline` cells of this run equal
those of the primary run exactly (largest absolute difference 0 for every measure M1 to M8, the occupied
neurons and the hotspot image counts). R1 (not met), R2 (met: −0.055, −0.101 to −0.008) and R5 (met: upper
end +0.54 below the 1.0 tolerance) therefore repeat the primary result. As section 2 says, R3 and R4 in
this run's `statements.json` compare with `augmented_msp`, not with random augmentation; they are S3 and S2
above.

Other measures: overall validation error is lower for `augmented_msp` than for `standard` (−0.22 points,
−0.26 to −0.18), the only interval for this pair that excludes 0; no measure differs clearly between
`augmented_som` and `augmented_msp`.
