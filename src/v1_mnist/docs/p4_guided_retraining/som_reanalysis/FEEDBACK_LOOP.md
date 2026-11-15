# The feedback loop: SOM analysis → targeted retraining → SOM structure

How the three steps of Phase 4 connect, and what the re-analysis found. Protocol: `SOM_REANALYSIS_PROTOCOL.md`. Numbers come from `outputs/v1_mnist/p4_guided_retraining/som_reanalysis/<run folder>/tables/`.

## The loop

1. **Analysis.** The Phase 2 SOM on the Phase 1 embeddings shows where the network makes its mistakes: a few neurons (the hotspots) hold a large share of the validation errors, and some neurons mix classes.
2. **Targeted retraining.** Training images that fall in those hotspot neurons, in neurons with purity below 0.95, or far from their neuron form the hard pool. The network is retrained with altered copies of only those images (`augmented_som`), and compared with the same recipe on random images (`augmented_random`) and with plain training (`standard`).
3. **New SOM structure.** The retrained network has a different embedding. A new SOM is trained on it with the Phase 2 recipe and measured with the same purity, hotspot and organisation measures.

## What retraining showed (test pass, already frozen)

Hotspot error was lower with `augmented_som` than with `standard` (−0.035616, interval −0.057534 to −0.013699) and than with `augmented_random` (−0.024658, interval −0.045205 to −0.006849), and no damage elsewhere was found (non-hotspot difference −0.000522, interval −0.001314 to +0.000232). The curriculum criteria were not met. These are the test-set hotspots of the Phase 2 map.

## What the re-analysis found

Results of run `som_reanalysis_som_15x15_nb11_ep250_s42_v2`. Validation images only; 5 seeds; mean paired difference (`augmented_som` minus the other arm) with a 95% interval.

| Statement | Measure | Result | Met? |
|---|---|---|---|
| R1 boundaries sharpen (vs `standard`) | weighted purity | +0.000076 (−0.001706 to +0.001859) | No |
| R2 hotspots shrink (vs `standard`) | hotspot error rate | −0.054524 (−0.100825 to −0.008224); lower in 5 of 5 seeds | Yes |
| R3 guidance sharpens (vs `augmented_random`) | weighted purity | +0.001162 (−0.001551 to +0.003875) | No |
| R4 guidance shrinks hotspots (vs `augmented_random`) | hotspot error rate | −0.031461 (−0.070453 to +0.007530) | No |
| R5 map not less organised (vs `standard`) | topographic error (points) | −0.285714 (−1.114630 to +0.543201); upper end is below the 1.0 tolerance | Yes |

Other measures (not statements). Low-purity share: +0.007714 vs `standard` and +0.003676 vs `augmented_random`; both intervals include 0. Class scatter index: +0.08 vs both arms; both intervals include 0. Share of errors in hotspots: +0.016331 vs `standard` (interval includes 0) and +0.062849 vs `augmented_random` (interval +0.035950 to +0.089749, larger in 5 of 5 seeds). Overall validation error: −0.001676 vs `standard` and −0.001124 vs `augmented_random`; both intervals exclude 0.

Mean over the 5 seeds, next to the Phase 2 reference map:

| Measure | Phase 2 map | `standard` | `augmented_som` | `augmented_random` |
|---|---|---|---|---|
| Weighted purity | 0.9819 | 0.9824 | 0.9824 | 0.9813 |
| Low-purity share | 0.0801 | 0.0718 | 0.0795 | 0.0758 |
| Class scatter index | 1.10 | 1.12 | 1.20 | 1.12 |
| Topographic error (%) | 6.18 | 7.04 | 6.76 | 6.66 |
| Hotspot error rate | 0.1954 | 0.2292 | 0.1746 | 0.2061 |
| Share of errors in hotspots | 0.3696 | 0.3669 | 0.3833 | 0.3204 |
| Overall validation error | 0.0088 | 0.0092 | 0.0075 | 0.0086 |

Linear CKA between the networks' embeddings (mean): 0.931 within `standard`, 0.917 within `augmented_random`, 0.893 within `augmented_som`, 0.910 between arms.

## How to read it

Two of the five statements were met (R2, R5) and three were not (R1, R3, R4).

- The hotspots shrank compared with plain training (R2). Against the random-image control the direction is the same, but the interval includes 0 (R4), so the data do not show that SOM-chosen images are better than random ones.
- The class boundaries did not become measurably sharper (R1, R3): purity is the same for all arms.
- The map stayed well organised (R5).
- Part of the hotspot drop comes from the network making fewer errors overall (overall error is lower for `augmented_som`), not only on the weak regions.
- The retrained `standard` networks have a higher hotspot error rate (0.2292) than the Phase 2 map (0.1954), so the Phase 2 map is not a fair baseline. The comparison between arms is the one to use.
- Hotspots are the worst neurons of each map and are chosen again for every map, so the hotspot error rate of a new map is not measured on the same images as before.
- Five seeds with one SOM seed can only show large effects.

## Limits

Five seeds, one SOM seed; arms chosen after the retraining test results were known; maps of different networks cannot be matched neuron by neuron. See protocol section 8.
