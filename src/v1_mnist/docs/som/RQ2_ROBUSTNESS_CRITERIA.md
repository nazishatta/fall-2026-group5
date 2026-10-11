# RQ2 error-geography criteria and robustness check

Written: 2026-10-06, before the 20x20 and 10x10 robustness results were viewed. Commit before running or opening them.
These rules are the project's standard RQ2 test: they are applied unchanged to every dataset (MNIST now; Fashion-MNIST, CIFAR-10 or others later).

## 1. What RQ2 claims, and where that comes from

From the project proposal (Post-Training Analysis and Novelty Detection Using NNSOM):
- Rationale, *error geography*: "Are errors concentrated in specific cluster regions, indicating a structural weakness?"
- Week 5: "error hotspots: SOM regions with **disproportionately high misclassification rates**".
- Objective 4: "overlap regions (sources of misclassification)" and "hard examples (samples near cluster boundaries)".
- Possible issue 4: neurons with few hits make "hotspot statistics unreliable".

The quantities are the same per-neuron ones NNSOM's post-training tools use: misclassification rate per neuron (`get_perc_misclassified`) and class mix per neuron (`cal_class_cluster_intersect`).

**Claim A: errors are concentrated in hotspots.**
**Claim B: errors occur where classes overlap.**
Distance to the best-matching neuron belongs to novelty detection (proposal Phase 3) and is reported as secondary only.

## 2. One baseline for everything: errors placed at random

Every claim is compared with what we would see if the CNN's errors were scattered at random over the validation images.
To build this baseline, the error/correct labels are shuffled across validation images (each image keeps its neuron and its class), and the same statistic is recomputed **200 times**.
A claim holds only if the real value is more extreme than the shuffled values (one-sided p < 0.05).
This needs no hand-picked threshold, so the rule means the same thing for any dataset, CNN or grid size.

## 3. Claim A: error hotspots (held out)

1. Split the validation images into two halves, each holding half of the errors and half of the correct images.
2. On half A, rank occupied neurons by the 95% Wilson lower bound of their error rate (ties: more errors, more images, lower index). The Wilson bound stops neurons with few hits from being picked by chance (proposal, issue 4). Take the top **k = max(5, 5% of neurons)** as hotspots (10x10: 5, 15x15: 11, 20x20: 20), so every grid uses the same share of its map.
3. On half B: **ratio = (hotspots' share of half-B errors) / (hotspots' share of half-B images)**. A ratio of 1 means no concentration.
4. Swap halves and repeat, over **100 random splits** (200 ratios). Statistic: the **median ratio**.
   Choosing hotspots on one half and measuring on the other stops finer grids from looking better by chance.

**Holds if both:** (a) the median ratio beats the shuffled-label baseline (p < 0.05), and (b) the 2.5th percentile of the 200 ratios is above 1 (concentration shows up consistently, not only in some splits).

## 4. Claim B: errors in overlap regions

Over occupied validation neurons, Spearman correlation of neuron error rate with class purity (expected **negative**) and with normalized class entropy (expected **positive**).

**Holds if both** correlations are beyond the shuffled-label baseline in the expected direction (each p < 0.05).

## 5. Minimum data

If the validation split has **fewer than 20 CNN errors**, the claims are reported as **not testable** for that dataset rather than judged.

## 6. Headline result and robustness verdict

- Claims are judged first on the **headline model** (for MNIST: `som_15x15_nb11_ep250_s42_v2`). Only claims that hold there are main RQ2 findings.
- Each **robustness model** (for MNIST: `som_20x20_nb15_ep250_s42_v2`, the grid the original protocol selected, and `som_10x10_nb7_ep250_s42_v2`, the grid the frozen rule selects on the surviving files) is then judged on those main claims:

| Main claims that also hold on the robustness model | Verdict |
|---|---|
| All | **RQ2 conclusion holds** |
| Some | **Holds in part** (name the failing claim) |
| None | **Does not hold** |
| Too few errors | **Not testable** |

## 7. Applying this to other datasets

Same code (`rq2_robustness.py`, `rq2_robustness_verdict.py`), same settings (200 shuffles, 100 splits, k = max(5, 5% of neurons), p < 0.05, at least 20 errors), same verdict table.
Per dataset, only the inputs change, and they are fixed **before** results are viewed: the frozen embeddings, the headline grid and the robustness grids (written in that dataset's config files).

## 8. Disclosure (MNIST)

While testing the code on 2026-10-06, the assistant that drafted these rules saw:
- for 20x20, the share of validation errors in the 10 neurons with the most errors, from the earlier RQ2 code (not a statistic used here);
- for the 15x15 headline model, the values of Claim A and Claim B under these rules (both clearly beyond chance).
No threshold above was chosen using those values: the rules use the shuffled-label baseline and conventional choices (p < 0.05, Wilson 95% bound) instead of tuned cut-offs. The team has not seen any 20x20 or 10x10 robustness result.

## 9. Reporting

For each model: number of validation errors; Claim A median ratio, its 2.5 to 97.5 percentile range, the shuffled-baseline median and 95th percentile, and p; Claim B correlations, their shuffled-baseline bounds and p-values; secondary distances; and the verdict.
Paper sentence: "Using criteria fixed in advance (RQ2_ROBUSTNESS_CRITERIA.md), the RQ2 conclusion [holds / holds in part / does not hold] on 20x20 and [ ... ] on 10x10."
