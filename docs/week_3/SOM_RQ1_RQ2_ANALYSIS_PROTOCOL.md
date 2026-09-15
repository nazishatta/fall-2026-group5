# Week 3 RQ1/RQ2 SOM Analysis Protocol

## Status

Frozen before examining BMU-level class structure or error geography
for the selected SOM.

This is a development-stage analysis protocol.

## Selected SOM

Grid: 20x20

Seed: 42

Artifact:

`outputs/week_3/som_models/som_20x20_seed42_gridstudy`

Expected SHA-256:

`528bb5e11cac077c155ba42855ed1c423aba7d0e70d86f5f42a0c7dee813c576`

The grid was selected using the previously frozen grid-selection
protocol.

## Data protection

RQ1 and RQ2 use only:

- frozen Week 2 training embeddings;
- frozen Week 2 validation embeddings.

The final test split must not be evaluated, mapped, inspected, or used
to modify this analysis.

All result metadata must record:

`test_evaluated = false`

## BMU assignment

BMUs are obtained directly from NNSOM `cluster_data()`.

For every train/validation sample record:

- original split-order index;
- frozen sample ID;
- ground-truth label;
- baseline prediction;
- baseline correctness;
- baseline confidence;
- BMU neuron index;
- SOM position coordinates;
- distance to its BMU prototype.

Every sample must be assigned exactly once.

The resulting assignment count must equal the input split size.

## Regression checks

The loaded selected SOM must reproduce the already frozen:

- train quantization error;
- validation quantization error;
- train occupied-neuron count;
- validation occupied-neuron count.

Failure of any regression check invalidates the analysis run.

## RQ1: representation structure

For every neuron and split report:

- sample support / hit count;
- class counts for digits 0 through 9;
- dominant true class;
- class purity;
- normalized Shannon entropy;
- mean BMU distance.

Purity is:

`maximum class count / neuron support`

Normalized entropy is Shannon entropy divided by `log(10)` so that
values are on [0, 1].

Empty neurons have undefined purity and entropy.

If multiple classes tie for dominant class, the lowest numeric class
returned by NumPy `argmax` is used as the deterministic tie-break.

Primary RQ1 visualizations:

1. training hit-count map;
2. training dominant-class map;
3. validation class-purity map;
4. validation normalized-entropy map.

RQ1 interpretation must report overlap/ambiguity rather than assuming
visual separation implies predictive quality.

## RQ2: error geography

RQ2 uses the validation split only.

For every neuron report:

- validation support;
- correct count;
- error count;
- error rate;
- mean confidence;
- mean confidence for correct samples;
- mean confidence for errors;
- mean BMU distance;
- mean BMU distance for correct samples;
- mean BMU distance for errors.

A 95% Wilson lower bound for the neuron error rate is also calculated
as a support-aware descriptive prioritization measure.

This is not treated as a multiple-comparison significance test.

Hotspot candidates are ranked prospectively by:

1. higher Wilson 95% lower bound;
2. higher raw error count;
3. higher support;
4. lower neuron index as deterministic final tie-break.

The top 20 are exported for inspection.

Primary RQ2 visualizations:

1. validation error-count map;
2. validation error-rate map;
3. validation Wilson-lower-bound map;
4. validation mean-confidence map.

Also report:

- fraction of all validation errors contained in the ten neurons with
  the largest raw error counts;
- descriptive Spearman correlations between neuron error rate and:
  normalized entropy, purity, and mean BMU distance.

These correlations are exploratory and are not treated as confirmatory
statistical evidence.

## Scientific interpretation

Maps alone do not establish a finding.

Every interpretation must consider neuron support.

Sparse high-error neurons must not be presented as equivalent to
well-supported error regions.

Metric disagreement must be reported.

No novelty/OOD threshold, OOD dataset, retraining decision, or final
test evaluation is permitted during this stage.

## Output isolation

Run ID:

`som_20x20_seed42_rq1_rq2_v1`

Outputs belong under:

`outputs/week_3/analysis/som_20x20_seed42_rq1_rq2_v1/`

Existing scientific outputs must not be overwritten.
