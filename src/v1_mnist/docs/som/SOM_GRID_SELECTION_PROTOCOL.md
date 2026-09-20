# Week 3 SOM Grid-Selection Protocol

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

- frozen Week 2 LeNet-5 embeddings;
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

`outputs/week_3/`

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

---

## Scientific Integrity Rule

The selected grid must follow this protocol even if another grid
produces a visually more attractive SOM.

No grid may be selected based on visualization quality, desired
class separation, downstream novelty-detection performance, or
preferred conclusions.
