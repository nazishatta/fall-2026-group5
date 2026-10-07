# SOM Grid-Selection Protocol (v1_mnist)

## Status

This protocol is frozen before evaluating the remaining 10x10 and
20x20 SOM candidates.

The existing 15x15 run is treated as an exploratory candidate and
remains immutable.

No final test-set SOM evaluation is permitted during grid selection.

---

## Scientific Objective

Select a SOM grid size for subsequent representation-structure and
error-geography analysis using training and validation data only.

The grid study is a development-stage model-selection experiment,
not final test evaluation.

---

## Candidate Grids

The predefined candidate set is:

- 10x10
- 15x15
- 20x20

The seed-42 runs form the primary grid-size comparison.

The existing frozen 15x15 seed-42 run is reused rather than rerun.

---

## Controlled Variables

Across grid candidates, the following must remain fixed:

- frozen LeNet-5 embeddings;
- feature layer and embedding dimensionality;
- sample ordering;
- train/validation split;
- preprocessing method;
- scaler fit on training data only;
- SOM implementation and NNSOM version;
- initialization procedure;
- economy-SVD memory-safe initialization;
- random seed for the primary grid study;
- training epochs;
- training steps;
- initial neighborhood setting;
- backend;
- evaluation implementation.

Grid dimensions are the intended experimental variable.

---

## Test-Set Protection

The final test split must not be used for:

- grid selection;
- SOM hyperparameter selection;
- feature-layer selection;
- preprocessing selection;
- threshold selection;
- candidate ranking.

The training pipeline must continue to record:

`test_evaluated = false`

during this study.

---

## Primary Selection Metrics

Grid selection uses validation-set:

1. Quantization Error (QE)
2. First-order Topological Error (TE1)
3. First+second-order Topological Error (TE1+2)

Lower values are better for all three metrics.

Occupancy is treated as a structural feasibility diagnostic rather
than an optimization target.

Training metrics are retained for diagnostics but do not determine
the winning grid.

---

## Occupancy Feasibility

A grid is considered structurally usable for this study when:

- training occupancy >= 0.85; and
- validation occupancy >= 0.75.

A grid failing either criterion is not preferred over a feasible grid
solely because it achieves a lower quantization error.

These thresholds are frozen before viewing the 10x10 and 20x20
results.

---

## Grid Ranking Rule

Among feasible candidates:

1. Rank validation QE from best to worst.
2. Rank validation TE1 from best to worst.
3. Rank validation TE1+2 from best to worst.
4. Sum the three ordinal ranks with equal weight.
5. Select the candidate with the lowest total rank.

This prevents selection based on a single favorable metric.

If two candidates have the same total rank, select the smaller grid
as the predefined parsimony tie-breaker.

No metric may be added, removed, or reweighted after observing the
10x10 or 20x20 results.

---

## Metric Disagreement

If QE and topology metrics favor different grids, the predefined
equal-weight rank rule is used.

The disagreement itself must also be reported because it may reveal
a trade-off between quantization fidelity and topology preservation.

---

## Reproducibility Check

After selecting the grid using the predefined seed-42 comparison,
the selected configuration should be checked using additional seeds
when computationally feasible.

Additional seeds are used to evaluate stability, not to cherry-pick
a better result.

If seed sensitivity materially changes the interpretation, the
instability must be reported.

---

## Artifact Rules

Every new scientific run must use a unique run_id.

Artifacts are isolated under:

`outputs/v1_mnist/som/`

with run-specific:

- SOM model;
- metrics JSON;
- console log;
- configuration snapshot;
- pre-run provenance;
- post-run provenance.

Existing scientific artifacts must never be silently overwritten.

---

## Frozen 15x15 Candidate

The existing 15x15 seed-42 run remains an immutable exploratory
candidate in the grid study.

Its previously verified artifacts and hashes must not be modified.

---

## Selection Outcome

This section must remain empty until the 10x10 and 20x20 runs have
completed.

After completion, record:

- candidate metrics;
- feasibility status;
- metric ranks;
- total rank;
- selected grid;
- rationale;
- any metric disagreement;
- reproducibility status.

### Recorded outcome (written 2026-10-06)

The rules above have not been changed. This section records what actually happened, in order. The rule breaks are listed one by one in `DEVIATIONS.md`.

**1. The rule picked 20x20 (2026-09-15, commit a4c4b0f).**
This table comes from `docs/week_3/SOM_GRID_SELECTION_RESULT.md`, which was later deleted and has been restored from git:

| Grid | Validation QE | QE rank | Validation TE1 (%) | TE1 rank | Validation TE1+2 (%) | TE1+2 rank | Validation occupancy | Rank sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10x10 | 1.342208 | 3 | 7.4381 | 2 | 1.1905 | 1 | 0.9900 | 6 |
| 15x15 | 1.132639 | 2 | 9.2476 | 3 | 1.6095 | 3 | 0.8889 | 8 |
| 20x20 | 0.968939 | 1 | 5.5333 | 1 | 1.3810 | 2 | 0.8150 | **4** |

**2. Those runs were then overwritten.**
Later the same day (commit ed8c4a8), all three runs were retrained under the **same run IDs**, which replaced the files behind the table above. This broke the rule that results must never be overwritten. On the new files, the same rule picks **10x10**:

| Grid | Validation QE | Validation TE1 (%) | Validation TE1+2 (%) | Validation occupancy | Rank sum |
|---|---:|---:|---:|---:|---:|
| 10x10 | 1.193277 | 4.8762 | 2.1619 | 1.0000 | **5** |
| 15x15 | 1.047913 | 7.5238 | 4.8000 | 0.9956 | 6 |
| 20x20 | 0.955526 | 8.9619 | 6.6762 | 0.9925 | 7 |

The surviving 20x20 model's fingerprint (SHA-256 `d3bf26cc...bb6c`) does not match the one recorded when 20x20 was picked (`528bb5e1...c576`). We report both tables rather than choosing one after the fact.

**3. We use 15x15 because the instructor asked for it.**
In the 2026-09-15 working session the instructor asked the team to try a 15x15 SOM, and confirmed this on issue #3 (2026-10-04). The switch to 15x15 (commit 18b5ae8, 2026-09-21) did not record this reason; it also deleted the result file above and removed the model fingerprint check. Both have since been restored.

**4. A fair test showed 15x15 is not a worse choice.**
We wrote the rules for a new test before running it (`PREREGISTRATION_grid_comparison.md`, commit abef97d), and gave every grid the same tuning (results in `GRID_COMPARISON_RESULTS.md`, commit cb48b99). The test asked how well each map's cells predict which new validation images the CNN gets wrong:
20x20 AUROC 0.630, 15x15 0.629; difference +0.001, 95% range −0.100 to +0.089. This is **inconclusive**: with 92 CNN mistakes, differences smaller than about 0.1 can't be detected.
20x20 fits the data more closely and has purer cells; 15x15 has lower topological error. Neither predicts the CNN's mistakes better.
Observation, not a finding: smaller grids tended to predict CNN errors slightly better on held-out data, likely because each cell holds more images. The differences were not statistically clear, and the RQ2 findings were the same on all three grids.

**5. Grids used in the report.**
- **Main results (RQ1, RQ2): 15x15** (neighbourhood 11, 250 epochs), as the instructor requested.
- **Robustness checks:** **20x20** (neighbourhood 15, 250 epochs), the grid this protocol first picked, and **10x10** (neighbourhood 7, 250 epochs), the grid the rule picks on the surviving files. Together with 15x15 these cover all three original candidates.
- **Result (2026-10-06):** using rules fixed before the check (`RQ2_ROBUSTNESS_CRITERIA.md`), the RQ2 findings **hold** on both 20x20 and 10x10. See `RQ2_ROBUSTNESS_RESULTS.md`.

Reproducibility: the protocol tables use seed 42; the fair test used 10 resampled maps per grid. All new runs use new run IDs, and each model's SHA-256 is recorded in a run manifest.

---

## Scientific Integrity Rule

The selected grid must follow this protocol even if another grid
produces a visually more attractive SOM.

No grid may be selected based on visualization quality, desired
class separation, downstream novelty-detection performance, or
preferred conclusions.
