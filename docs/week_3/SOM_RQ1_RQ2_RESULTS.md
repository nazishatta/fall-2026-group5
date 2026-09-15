# Week 3 RQ1/RQ2 Results

## Status

Frozen development-stage results for the selected 20x20 SOM.

Run ID:

`som_20x20_seed42_rq1_rq2_v1`

Selected SOM:

`outputs/week_3/som_models/som_20x20_seed42_gridstudy`

Model SHA-256:

`528bb5e11cac077c155ba42855ed1c423aba7d0e70d86f5f42a0c7dee813c576`

Analysis code commit:

`5e736ed36fd04c44c2d8dceaa98e4bcbe0caef99`

NNSOM version:

`1.8.3`

Final test evaluated:

**No**

## RQ1 — Representation Structure

### Training

- Occupied neurons: 354/400
- Occupancy: 0.8850
- Sample-weighted class purity: 0.985918
- Sample-weighted normalized entropy: 0.019193

### Validation

- Occupied neurons: 326/400
- Occupancy: 0.8150
- Sample-weighted class purity: 0.978762
- Sample-weighted normalized entropy: 0.027479

### RQ1 Interpretation

The selected SOM exhibits strong class-structured organization:
validation purity is high and normalized class entropy is low.

This does not imply perfect separation. Mixed neurons remain and are
important for the error-geography analysis.

These are descriptive development-stage results from a single selected
seed and must not be interpreted as evidence of statistical stability
across seeds.

## RQ2 — Error Geography

Validation samples:

10500

Validation errors:

93

Validation error rate:

0.008857

Errors contained in the ten neurons with the largest raw error counts:

41/93
(0.4409)

### Sample-level BMU distance

- Mean distance, correct: 1.217743
- Mean distance, error: 1.466475
- Median distance, correct: 1.177700
- Median distance, error: 1.393465

### Baseline confidence

- Mean confidence, correct: 0.997036
- Mean confidence, error: 0.813575

### Exploratory neuron-level associations

- Error rate vs normalized class entropy:
  Spearman rho = 0.741762
- Error rate vs purity:
  Spearman rho = -0.736987
- Error rate vs mean BMU distance:
  Spearman rho = 0.349297

These correlations are exploratory descriptive statistics rather than
confirmatory significance tests.

## Main Evidence

The SOM reveals non-uniform error geography.

Although the baseline makes relatively few validation errors overall,
44.1% of those errors occur in only ten neurons when neurons are ranked
by raw error count.

Misclassified samples have larger BMU distances on average than
correctly classified samples, which is consistent with the hypothesis
that quantization-related signals may contain information about
difficult in-distribution samples.

However, neuron-level error rate is more strongly associated with
class entropy and purity than with mean BMU distance in this run.
Therefore, the current evidence does not justify claiming that
quantization distance alone explains or detects model failures.

## Support Caveat

Several high-error-rate neurons have very small support.

The predefined Wilson-lower-bound ranking reduces but does not remove
this issue. Hotspots must therefore always be interpreted jointly with
sample support and raw error counts.

Well-supported regions are more defensible targets for later sample
inspection and intervention than isolated sparse neurons.

## Leakage / Integrity

- Grid selection used training and validation only.
- RQ1/RQ2 use training and validation only.
- Final test data were not mapped or inspected for these analyses.
- No OOD threshold or retraining decision was selected from test data.
- All BMU assignments reproduced the frozen selected-SOM QE and
  occupancy metrics before results were accepted.

## Current Conclusion

RQ1 receives descriptive support: the learned feature representation is
strongly class-organized in SOM space while retaining localized overlap.

RQ2 also receives descriptive support: baseline errors are spatially
non-uniform and concentrated in particular SOM regions.

The evidence additionally suggests that misclassified validation
samples tend to have larger prototype distances, but class mixing is a
stronger neuron-level correlate of error rate in this development run.

These conclusions remain conditional on the selected grid and seed and
require later stability analysis before publication-level claims.

## Next Stage

The Week 3 representation and error-geography results are now frozen.

The next scientific stage is RQ3 novelty/OOD detection, with thresholds
and OOD evaluation protocols defined before final evaluation.
