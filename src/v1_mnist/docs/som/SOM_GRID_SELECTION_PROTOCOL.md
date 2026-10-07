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

The rules above are unchanged. This record was written after the events below, and states them as they happened.

**1. Result of this protocol (2026-09-15, commit a4c4b0f).**
Applying the rank-sum rule selected **20x20**. Original table (from the deleted `docs/week_3/SOM_GRID_SELECTION_RESULT.md`, restored from git):

| Grid | Validation QE | QE rank | Validation TE1 (%) | TE1 rank | Validation TE1+2 (%) | TE1+2 rank | Validation occupancy | Rank sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10x10 | 1.342208 | 3 | 7.4381 | 2 | 1.1905 | 1 | 0.9900 | 6 |
| 15x15 | 1.132639 | 2 | 9.2476 | 3 | 1.6095 | 3 | 0.8889 | 8 |
| 20x20 | 0.968939 | 1 | 5.5333 | 1 | 1.3810 | 2 | 0.8150 | **4** |

**2. The candidate runs were overwritten.**
Later the same day (commit ed8c4a8), the three candidate runs were retrained under the **same run IDs** with the batch-SOM trainer, replacing the artifacts behind the table above. This broke the rule "existing scientific artifacts must never be silently overwritten". On the surviving files the same rule selects **10x10**:

| Grid | Validation QE | Validation TE1 (%) | Validation TE1+2 (%) | Validation occupancy | Rank sum |
|---|---:|---:|---:|---:|---:|
| 10x10 | 1.193277 | 4.8762 | 2.1619 | 1.0000 | **5** |
| 15x15 | 1.047913 | 7.5238 | 4.8000 | 0.9956 | 6 |
| 20x20 | 0.955526 | 8.9619 | 6.6762 | 0.9925 | 7 |

The surviving 20x20 model's SHA-256 (`d3bf26cc...bb6c`) does not match the hash recorded when 20x20 was selected (`528bb5e1...c576`). We report both tables rather than choose between them after the fact.

**3. 15x15 is used as the instructor-requested experiment.**
In the 2026-09-15 working session the instructor asked the team to use a 15x15 SOM as an experiment; the instructor confirmed this on issue #3 (2026-10-04). Commit 18b5ae8 (2026-09-21) switched the analysis to 15x15 but did not record the reason, deleted the result document above, and removed the model hash check. These are listed in `DEVIATIONS.md`.

**4. Pre-registered check that 15x15 does not disadvantage the analysis.**
A new comparison was pre-registered before any run (`PREREGISTRATION_grid_comparison.md`, commit abef97d) and run with an equal tuning budget for every grid (results: `GRID_COMPARISON_RESULTS.md`, commit cb48b99). Each map's cells were used to predict which held-out validation images the CNN gets wrong:
20x20 AUROC 0.630 vs 15x15 0.629; difference +0.001, 95% interval -0.100 to +0.089 (**inconclusive**; with 92 validation errors, differences below about 0.1 cannot be resolved).
20x20 gave lower quantization error and purer cells; 15x15 gave lower topological error. Neither predicted CNN errors better.

**5. Grids used for reporting.**
- **Headline results (RQ1, RQ2): 15x15** (init_neighborhood 11, 250 epochs), the instructor-requested configuration.
- **Robustness checks: 20x20** (init_neighborhood 15, 250 epochs), the grid originally selected by this protocol, **and 10x10** (init_neighborhood 7, 250 epochs), the grid the rule selects on the surviving files. Together with 15x15 these complete the original 10x10 / 15x15 / 20x20 ablation. RQ2 is repeated on both, and we state whether the conclusion holds on each.
- **Result (2026-10-06):** under criteria committed before the runs (`RQ2_ROBUSTNESS_CRITERIA.md`), the RQ2 conclusion **holds** on both 20x20 and 10x10. See `RQ2_ROBUSTNESS_RESULTS.md`.

Reproducibility: results above use seed 42 for the protocol tables and 10 bootstrap-resampled maps per grid for the pre-registered comparison. All new runs use new run IDs with model SHA-256 recorded in a run manifest.

---

## Scientific Integrity Rule

The selected grid must follow this protocol even if another grid
produces a visually more attractive SOM.

No grid may be selected based on visualization quality, desired
class separation, downstream novelty-detection performance, or
preferred conclusions.
