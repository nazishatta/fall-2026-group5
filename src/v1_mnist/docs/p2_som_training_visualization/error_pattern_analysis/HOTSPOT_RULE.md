# Error hotspot rule (Week 5: Error Pattern Analysis)

Written on 2026-10-10, before any Week 4–5 results were produced with the merged code. It does not create a new rule: it adopts, for all of Week 5, the hotspot rule that was already fixed in advance for the robustness check.

## The rule

A neuron is an **error hotspot** if it is one of the top **k** occupied neurons when all occupied neurons are ranked by the **95% Wilson lower bound** of their validation error rate.

- Ties are broken by more errors, then more images, then the lower neuron index.
- **k = max(5, round(5% of the neurons))**: 5 on 10x10, 11 on 15x15, 20 on 20x20.
- Only the validation split is used; the test split never is.

Code: `src/v1_mnist/component/utils/hotspots.py` (`n_hotspots`, `top_hotspots`). The same code is used in two places:

| Where | What it does with the rule |
|---|---|
| `error_pattern_analysis/error_patterns.py` | Applies it to the full validation split to list the hotspots (tables, hotspot images, Week 5 heatmaps). |
| `error_pattern_analysis/error_pattern_robustness.py` | Applies it to half of the validation images and measures the other half (the held-out test). |

## Where it comes from

- **Project proposal**, Week 5: "Identify error hotspots: SOM regions with disproportionately high misclassification rates." Possible issue 4: neurons with few hits make "hotspot statistics unreliable"; the Wilson lower bound keeps small neurons from being picked by chance.
- **`ERROR_PATTERN_ROBUSTNESS_CRITERIA.md`**, section 3, step 2: exactly this rule, written down on 2026-10-06 (commit `d9604e0`) before any robustness result was viewed. That file is never edited; there the rule belongs to "Claim A" (hotspot concentration).

## Why there is now one rule

Until 2026-10-10 the code used three different hotspot rules:

| Rule | Where | Hotspots on the 15x15 map |
|---|---|---|
| A: top 20 by Wilson lower bound | `cluster_analysis/bmu_analysis.py` (now retired) | 20 neurons, 52 of 92 errors |
| B: at least 20 images and Wilson lower bound above the overall error rate | `error_pattern_analysis/error_patterns.py` (before the merge) | 9 neurons, 27 of 92 errors |
| C: this rule | `error_pattern_analysis/error_pattern_robustness.py` | 11 neurons, 34 of 92 errors |

Different tables would have described different hotspots, and Week 9 (hard example mining) needs one list to start from. Rule C was chosen because it is the **only one fixed in advance and statistically tested**, not because of the list it produces.

## How to report hotspot numbers

- **The concentration result to quote is the held-out one**: on 15x15 the hotspots hold a median **11.1 times** their share of the errors on images not used to choose them (`ERROR_PATTERN_ROBUSTNESS_RESULTS.md`; 10.5 to 13.4 times across the three grids).
- Counting errors inside the hotspots on the **same** images used to choose them overstates the concentration (on 15x15: 34 of 92 errors in 174 of 10,500 images, about 22 times). `error_patterns.py` reports these counts as "descriptive" only.

## What had already been seen when this was written

- **Rule B's results** are in the Week 5 LaTeX report (`reports/Latex_report/v1_mnist/error_patterns/`, 2026-10-03): 9 hotspots (neurons 22, 67, 73, 86, 101, 137, 139, 145, 220) with 27 of 92 errors. That report is left unchanged; it describes the old rule.
- On **2026-10-09**, while planning the merge, the three rules were compared once on the 15x15 validation split (read-only, nothing saved): the counts in the table above. Rule C picks neurons 37, 47, 67, 86, 109, 124, 130, 154, 160, 176 and 213; only 67 and 86 are also in rule B's list.
- The decision to use rule C everywhere was made on 2026-10-09 for the reasons above.
- On **2026-10-10** the merged code was checked in a scratch folder. It gave the same 11 neurons (in rank order: 130, 124, 47, 67, 37, 154, 86, 109, 160, 213, 176), 92 validation errors and validation purity 0.9819. The robustness results were unchanged by the switch to the shared code.

## Not used

The configs contain `visualization.hotspot_error_threshold: 0.35`. No code reads it; it is not part of any hotspot rule.
