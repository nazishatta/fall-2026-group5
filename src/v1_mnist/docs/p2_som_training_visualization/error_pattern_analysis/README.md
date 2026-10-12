# Week 5: error-pattern robustness check (MNIST)

Phase 2 (SOM Training & Visualization), Week 5 (Error Pattern Analysis). The proposal asks us to "identify error hotspots: SOM regions with disproportionately high misclassification rates" (Week 5) and to find "overlap regions (sources of misclassification)" (Objective 4).

## What this folder is about

Our main SOM is 15x15 (why: `../som_nnsom_training/README.md`). This folder answers: **do our findings about the CNN's mistakes stay the same on other grid sizes (20x20 and 10x10)?**

## Read the files in this order

| # | File | What it is |
|---|---|---|
| 1 | `ERROR_PATTERN_ROBUSTNESS_CRITERIA.md` | The rules for checking whether our error-pattern findings also hold on 20x20 and 10x10, written **before** looking. These rules are meant to be reused for future datasets. |
| 2 | `ERROR_PATTERN_ROBUSTNESS_RESULTS.md` | Results of that check. |
| 3 | `HOTSPOT_RULE.md` | The one rule for which neurons count as error hotspots, used everywhere in Week 5 since 2026-10-10 (it is the rule fixed in advance in file 1), and how to report hotspot numbers. |

> Files 1 and 2 were renamed on 2026-10-09 from `RQ2_ROBUSTNESS_CRITERIA.md` and `RQ2_ROBUSTNESS_RESULTS.md`, to use the project proposal's term (Error Pattern Analysis) instead of "RQ2", and moved here from `docs/som/` on 2026-10-10. The text of file 1 is untouched (see below), so it still says "RQ2", "Claim A" (= hotspot concentration) and "Claim B" (= overlap regions).

## Why file 1 is never edited

File 1 was written **before** the results. It is our evidence that we did not change the rules after seeing the results, so we leave it exactly as it was, even for wording. It is explained in plain words in file 2.

## Words you will see

- **Error hotspot:** a cell where the CNN makes many more mistakes than its share of images would suggest. Exactly which cells count is set by one rule (`HOTSPOT_RULE.md`): the top 5% of cells by Wilson lower bound (11 cells on 15x15).
- **Wilson lower bound:** a cautious estimate of a cell's error rate that stays low when the cell has only a few images, so tiny cells are not called hotspots by chance.
- **Overlap region:** a cell where images of different digits mix.
- **Headline model / robustness check:** the main model we report (15x15), and extra models (20x20, 10x10) used only to check that the results don't depend on that choice.
- **Shuffled baseline:** the same numbers recomputed after randomly reassigning which images are mistakes; a finding is real only if it beats this.

## Final result

- **The error-pattern findings hold on all three grids (15x15, 20x20, 10x10):** about 5% of the cells hold 10 to 13 times their fair share of the CNN's mistakes (measured on held-out images), and mistakes are most common in cells where different digits mix. Both findings are far beyond chance (p = 0.005).
