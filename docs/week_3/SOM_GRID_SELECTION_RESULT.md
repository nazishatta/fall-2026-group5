# Week 3 SOM Grid-Selection Result

## Decision

**Selected development SOM: 20x20, seed 42.**

Selection followed the grid-selection protocol frozen before the
10x10 and 20x20 results were observed.

The final test split was not evaluated or used in this decision.

## Quantitative comparison

| Grid | Validation QE | QE rank | Validation TE1 (%) | TE1 rank | Validation TE1+2 (%) | TE1+2 rank | Validation occupancy | Rank sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10x10 | 1.342208 | 3 | 7.4381 | 2 | 1.1905 | 1 | 0.9900 | 6 |
| 15x15 | 1.132639 | 2 | 9.2476 | 3 | 1.6095 | 3 | 0.8889 | 8 |
| 20x20 | 0.968939 | 1 | 5.5333 | 1 | 1.3810 | 2 | 0.8150 | **4** |

## Interpretation

The metrics disagree rather than uniformly favoring one grid.

The 20x20 candidate has the lowest validation quantization error and
the lowest first-order topological error.

The 10x10 candidate has the lowest first+second-order topological
error and the highest validation occupancy.

Following the prospectively frozen equal-weight rank-sum rule, 20x20
has the lowest total rank and is therefore selected.

The occupancy reduction at 20x20 is retained as a structural
trade-off to report in subsequent analysis rather than being hidden
or used post hoc to change the selection rule.

## Selected artifact

`outputs/week_3/som_models/som_20x20_seed42_gridstudy`

## Next stage

Use the selected SOM for BMU assignment, RQ1 representation analysis,
and RQ2 error-geography analysis.

OOD/novelty evaluation remains out of scope until the Week 3
representation and error-geography analyses are frozen.
