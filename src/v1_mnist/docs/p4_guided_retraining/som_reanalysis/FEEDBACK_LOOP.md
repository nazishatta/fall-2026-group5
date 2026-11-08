# The feedback loop: SOM analysis → targeted retraining → SOM structure

How the three steps of Phase 4 connect, and what the re-analysis found. Protocol: `SOM_REANALYSIS_PROTOCOL.md`. Numbers come from `outputs/v1_mnist/p4_guided_retraining/som_reanalysis/<run folder>/tables/`.

## The loop

1. **Analysis.** The Phase 2 SOM on the Phase 1 embeddings shows where the network makes its mistakes: a few neurons (the hotspots) hold a large share of the validation errors, and some neurons mix classes.
2. **Targeted retraining.** Training images that fall in those hotspot neurons, in neurons with purity below 0.95, or far from their neuron form the hard pool. The network is retrained with altered copies of only those images (`augmented_som`), and compared with the same recipe on random images (`augmented_random`) and with plain training (`standard`).
3. **New SOM structure.** The retrained network has a different embedding. A new SOM is trained on it with the Phase 2 recipe and measured with the same purity, hotspot and organisation measures.

## What retraining showed (test pass, already frozen)

Hotspot error was lower with `augmented_som` than with `standard` (−0.035616, interval −0.057534 to −0.013699) and than with `augmented_random` (−0.024658, interval −0.045205 to −0.006849), and no damage elsewhere was found (non-hotspot difference −0.000522, interval −0.001314 to +0.000232). The curriculum criteria were not met. These are the test-set hotspots of the Phase 2 map.

## What the re-analysis found

Results of run `[TBD: run folder]`. Validation images only; 5 seeds; mean paired difference with 95% interval.

| Statement | Measure | Result |
|---|---|---|
| R1 boundaries sharpen (vs `standard`) | weighted purity | [TBD] |
| R2 hotspots shrink (vs `standard`) | hotspot error rate | [TBD] |
| R3 guidance sharpens (vs `augmented_random`) | weighted purity | [TBD] |
| R4 guidance shrinks hotspots (vs `augmented_random`) | hotspot error rate | [TBD] |
| R5 map not less organised (vs `standard`) | topographic error | [TBD] |

Class scatter index, low-purity share and CKA: [TBD]. Phase 2 reference map values: [TBD].

## How to read it

[TBD: after the run. Say plainly which statements were met, which were not, and that 5 seeds with one SOM seed can only show large effects.]

## Limits

Five seeds, one SOM seed; arms chosen after the retraining test results were known; maps of different networks cannot be matched neuron by neuron. See protocol section 8.
