# Results: guided retraining on Fashion-MNIST (Phase 4)

The recorded outcome of [`RETRAINING_PROTOCOL.md`](RETRAINING_PROTOCOL.md). It is kept in this file, not in
the protocol's section 5, so the stamped protocol stays byte-identical and its stamp
(`retraining_protocol_stamp.txt`, 2026-11-22T07:17:44Z) can still be checked, as for MNIST. Every number is
copied from the result file named under each table; MNIST's values are from the same files of its own runs.
The test split was read once, by the `final_test` stage, after the stamp.

Settings that differ from MNIST, all set before the test pass and recorded in `../DEVIATIONS.md`:
augmentation ranges halved twice (revision r3: rotation 2.5°, shift 0.25 px, zoom 0.975–1.025, shear 2°)
because MNIST's ranges failed the transform check; macro-F1 guard 0.0033 (MNIST 0.0016), measured by its rule.
The SOM pool holds 19,997 of the 49,000 training images, 41% (MNIST 4,217, 8.6%), because the same purity rule catches
more neurons on the less pure Fashion-MNIST map. The test split has 422 hotspot images (MNIST 146).

## 1. Test error by arm (mean over seeds 1-5)

| Arm | Overall | Hotspot | Outside hotspots | Macro-F1 |
|---|---|---|---|---|
| `standard` | 0.0934 | 0.3597 | 0.0822 | 0.9062 |
| `augmented_som` | 0.0938 | 0.3398 | 0.0835 | 0.9061 |
| `augmented_random` | 0.0940 | 0.3502 | 0.0833 | 0.9058 |
| `augmented_msp` | 0.0943 | 0.3521 | 0.0835 | 0.9053 |
| `curriculum_som` | 0.1263 | 0.5185 | | |
| `curriculum_random` | 0.1085 | 0.4209 | | |

Sources: `test_results.json` of `targeted_augmentation/augmentation_comparison_som_15x15_nb11_ep250_s42_r3/`
and `hard_example_mining/curriculum_comparison_som_15x15_nb11_ep250_s42/` (under `outputs/v1_fashion_mnist/p4_guided_retraining/`);
all other arms and comparisons are in the same files.

## 2. Success statements (protocol section 2, MNIST protocol section 5)

Difference of the seed-averaged test error, first arm minus second; 95% paired bootstrap interval over test
images (2,000 resamples, seed 2026).

| Statement | Comparison | Fashion-MNIST | Verdict | MNIST | MNIST verdict |
|---|---|---|---|---|---|
| **A1 (primary)** | `augmented_som` − `augmented_random`, hotspot error | −0.0104 [−0.0322, +0.0114]; guard holds | **not met** (interval includes 0) | −0.0247 [−0.0452, −0.0068] | met |
| A2 | `augmented_som` − `standard`, hotspot error | −0.0199 [−0.0479, +0.0071]; guard holds | not met | −0.0356 [−0.0575, −0.0137] | met |
| A3 | `augmented_som` − `standard`, error outside hotspots; upper end ≤ 0.002 | +0.0013 [−0.0012, +0.0037] | not met (upper end 0.0037) | −0.0005 [−0.0013, +0.0002] | met |
| C1 | `curriculum_som` − `standard`, overall error | +0.0329 [+0.0293, +0.0364]; guard fails (macro-F1 −0.034) | not met: curriculum is worse | +0.0045 [+0.0033, +0.0057] | not met: worse |
| C2 | `curriculum_som` − `curriculum_random`, hotspot error | +0.0976 [+0.0664, +0.1284]; guard fails | not met: SOM curriculum is worse | +0.0918 [+0.0548, +0.1302] | not met: worse |

## 3. SOM re-analysis (validation images, new maps; `SOM_REANALYSIS_PROTOCOL.md`, `MSP_ARM_ADDENDUM.md`)

Per-seed differences, mean and 95% paired t-interval over the 5 seeds.

| Statement | Comparison | Fashion-MNIST | Verdict | MNIST | MNIST verdict |
|---|---|---|---|---|---|
| R1: boundaries sharpen | `augmented_som` − `standard`, weighted purity | −0.0087 [−0.0182, +0.0008] | not met | +0.0001 [−0.0017, +0.0019] | not met |
| R2: hotspots shrink | `augmented_som` − `standard`, hotspot error rate | −0.0009 [−0.0225, +0.0207] | not met | −0.0545 [−0.1008, −0.0082] | met |
| R3: guidance sharpens | `augmented_som` − `augmented_random`, weighted purity | −0.0048 [−0.0125, +0.0029] | not met | +0.0012 [−0.0016, +0.0039] | not met |
| R4: guidance shrinks hotspots | `augmented_som` − `augmented_random`, hotspot error rate | −0.0319 [−0.0516, −0.0122] | **met** | −0.0315 [−0.0705, +0.0075] | not met |
| R5: map not less organised | `augmented_som` − `standard`, topographic error (points); upper end ≤ 1.0 | +0.81 [−0.16, +1.79] | not met (upper end 1.79) | −0.29 [−1.11, +0.54] | met |
| S1: confidence guidance shrinks hotspots | `augmented_msp` − `standard`, hotspot error rate | +0.0065 [−0.0313, +0.0443] | not met | −0.036 [−0.093, +0.021] | not met |
| S2 (= R4 of the secondary run) | `augmented_som` − `augmented_msp`, hotspot error rate | −0.0074 [−0.0564, +0.0416] | not met | −0.0184 [−0.0762, +0.0394] | not met |

Sources: `tables/statements.json` and `tables/paired_comparisons.json` of
`som_reanalysis/som_reanalysis_som_15x15_nb11_ep250_s42/` and `som_reanalysis/som_reanalysis_msp_arm_som_15x15_nb11_ep250_s42/`.

## 4. Novelty after retraining (SVHN; `NOVELTY_AFTER_RETRAINING_PROTOCOL.md`)

QE AUROC on Fashion-MNIST validation (normal) against SVHN test (novel), mean over seeds: `standard` 0.838,
`augmented_som` 0.715, `augmented_random` 0.744 (MNIST: 0.976, 0.984, 0.983). Max-softmax: 0.875, 0.852, 0.884.

| Statement | Fashion-MNIST | Verdict | MNIST verdict |
|---|---|---|---|
| N1: the SOM still detects novelty (every network's QE AUROC above 0.5) | no network at or below 0.5 | shown | shown |
| N2: guided retraining does not harm (`augmented_som` − `standard`, QE AUROC, margin 0.01) | −0.123 [−0.151, −0.094] | **opposite shown** | inconclusive (+0.008 [−0.014, +0.030]) |
| N3: guidance does not harm (`augmented_som` − `augmented_random`) | −0.029 [−0.179, +0.121] | inconclusive | inconclusive |
| N4: the gap to max-softmax does not widen (QE minus max-softmax, `augmented_som` − `standard`) | −0.100 [−0.165, −0.035] | **opposite shown** | inconclusive |

Source: `novelty_after_retraining/novelty_after_retraining_som_15x15_nb11_ep250_s42/results.json`.

## 5. Reading

- **The primary statement A1 is not met.** On Fashion-MNIST, adding altered copies of the SOM-chosen images
  lowered the hotspot test error by 1.0 point more than adding copies of random images, but the interval
  includes 0. On MNIST the same comparison was met (2.5 points). The failed criterion is the result; the
  criteria are not changed.
- **Two declared conditions limit this test.** The augmentation had to be made four times weaker than on
  MNIST to keep the altered images readable for the network (`../DEVIATIONS.md`), which leaves it less room to
  help; and the SOM pool covers 41% of the training images (MNIST 8.6%), so the SOM and random pools overlap more
  than on MNIST: a random pool of the same size and class mix shares more of its images with the SOM pool by chance.
  Both were fixed before the test split was read.
- **Augmentation did not measurably change overall error**, but damage outside the hotspots of up to 0.4
  points cannot be ruled out (A3 not met).
- **The curriculum hurts on both datasets**, more on Fashion-MNIST (3.3 points overall), and the SOM-chosen
  curriculum is worse than the random one. The schedule (MNIST's, unchanged) lets in 10% of the hard pool and
  10% more every 5 epochs, so in 20 epochs it reaches only 40%: the hardest 60% of the pool is never trained on.
  With the Fashion-MNIST pool at 41% of the training set, that is about a quarter of all training images, against
  about 5% on MNIST (pool 8.6%), which fits the larger loss here.
- **On the new maps**, SOM guidance shrinks the weak regions compared with random guidance (R4 met), but not
  compared with standard training (R2 not met); on MNIST it was the other way round. The new maps may be less
  well organised (R5 not met).
- **Novelty detection got worse after retraining.** Both augmented arms lower the SOM's SVHN AUROC
  (0.838 to 0.715 and 0.744) and the gap to max-softmax widens (N2 and N4: opposite shown). On MNIST
  retraining left it unchanged. The `standard` value (0.838) lies within the spread seen on the Phase 1 network
  with SOM seeds 1-5 alone (0.766–0.884, Phase 3 protocol section 7), so single values move a lot; the
  comparisons above are paired by seed for that reason.
