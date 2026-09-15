# Week 3 SOM Grid-Selection Protocol

## Status

Frozen before observing the 10x10 and 20x20 candidate results.

The 15x15 candidate was already explored before this protocol was
written. Therefore, this document is not a full preregistration.
It prospectively freezes the comparison and selection rule before
evaluating the remaining unseen grid candidates.

## Objective

Select one SOM grid for subsequent representation analysis and
error-geography analysis using development data only.

The final test split must not be used for grid selection.

## Candidate grids

The predefined grid candidates are:

- 10x10
- 15x15
- 20x20

The primary comparison uses random seed 42.

The existing frozen 15x15 seed-42 result is reused and must not be
rerun or modified.

## Controlled variables

The following remain fixed across candidates:

- frozen Week 2 LeNet-5 embeddings;
- 84-dimensional feature representation;
- train/validation split;
- preprocessing;
- scaler fitted on training data only;
- NNSOM implementation/version;
- economy-SVD initialization workaround;
- random seed 42;
- training epochs;
- training steps;
- initialization neighborhood;
- backend;
- evaluation implementation.

Only SOM grid dimensions change.

## Test protection

The test split must not influence:

- grid selection;
- preprocessing;
- feature choice;
- SOM hyperparameters;
- candidate ranking;
- downstream threshold selection.

All grid-study metric files must record:

`test_evaluated = false`

## Selection metrics

Grid selection uses validation:

1. Quantization Error (QE)
2. First-order Topological Error (TE1)
3. First+second-order Topological Error (TE1+2)

Lower is better for all three.

Occupancy is reported as a structural diagnostic but is not optimized
directly.

Training metrics are diagnostic and do not determine the selected grid.

## Predefined ranking rule

For the three candidate grids:

1. rank validation QE from lowest to highest;
2. rank validation TE1 from lowest to highest;
3. rank validation TE1+2 from lowest to highest;
4. assign equal weight to all three ranks;
5. sum the three ranks;
6. select the grid with the lowest total rank.

If total ranks tie, select the smaller grid as the predefined
parsimony tie-breaker.

Metric weights, candidate grids, or tie-breaking rules must not be
changed after observing the 10x10 or 20x20 results.

## Metric disagreement

If QE and topology metrics prefer different grids, the predefined
rank-sum rule determines selection.

The disagreement must also be reported rather than hidden.

## Occupancy

Training and validation occupancy must be reported for every grid.

Occupancy is used to diagnose under-utilization or fragmentation of
the SOM, but it does not override the predefined rank rule after
results are observed.

## Reproducibility

The seed-42 comparison determines the development-stage grid choice.

The selected configuration should later receive additional-seed
stability analysis for publication-quality reporting. Additional
seeds must assess stability rather than select a favorable run.

## Artifact isolation

Every new scientific run must have a unique run_id.

All experiment artifacts remain under:

`outputs/week_3/`

Existing models, metrics, logs, configuration snapshots, and
provenance records must not be silently overwritten.

## Scientific integrity

Grid selection must not depend on:

- visual attractiveness;
- desired class separation;
- later novelty/OOD performance;
- downstream intervention performance;
- preferred conclusions.

The predefined quantitative rule determines the grid.
