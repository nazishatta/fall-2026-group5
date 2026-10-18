# QE Random-Control Reproduction

## Scope

Validation-only reproduction of the QE error-enrichment result.
No MNIST final-test data or external OOD datasets were used.

The historical result remains unchanged:

`outputs/v1_mnist/p3_novelty_detection/qe_novelty_detection/qe_random_control_evaluation_initial/qe_random_control_metrics.json`

## Reproduction command

From the repository root:

    python -m src.v1_mnist.component.pipeline.p3_novelty_detection.run_qe_random_control

The command is read-only by default.

Configuration:
`src/v1_mnist/component/configs/qe_random_control.yaml`

The `--save` option creates a new, uniquely named run directory.
It must not overwrite an existing run.

## Frozen-input verification

The generator verifies the validation input SHA-256 hashes against
the frozen QE summary.

Validation BMU-distance SHA-256:
`b0983bc8dba74998195d88011eabd1431bec449958932203c2beb3d4ec136186`

Validation CNN-correctness SHA-256:
`708271197be47a028bb3b87cd5afaa813fbec83c68ad13553c2b142e11147edf`

## Reproduced statistics

| Measurement | Historical | Recomputed |
| --- | ---: | ---: |
| Validation samples | 10500 | 10500 |
| QE-flagged samples | 539 | 539 |
| CNN errors | 92 | 92 |
| Captured CNN errors | 22 | 22 |
| Error capture rate | 0.2391304348 | 0.2391304348 |
| Expected random errors | 4.7226666667 | 4.7226666667 |
| Exact one-sided p-value | 8.83430658e-10 | 8.83430658e-10 |

The Wilson 95% confidence interval also reproduced:
`[0.16355145893303868, 0.33562143982467696]`.

## Historical seeded-control discrepancy

The original report records seeds `[11, 23, 37, 47, 59]`
and captured-error counts `[7, 4, 5, 7, 7]`, with mean 6.0.

The new generator explicitly samples using
`numpy.random.default_rng(seed).choice(n, size=k, replace=False)`.

Using the verified validation arrays, it produced
`[6, 5, 5, 5, 5]`, with mean 5.2.

The original report states that its seed-specific counts were
not recomputed when that report was saved. Its original
sampling procedure has not been established.

Therefore, only the aggregate statistics are claimed to have
been independently reproduced. The five historical seeded
outcomes are not claimed to have been reproduced.

## Leakage control

The novelty threshold was calculated using training data.
The comparison uses MNIST validation QE distances and CNN
correctness only.

No MNIST final-test examples or external OOD examples were
accessed during this reproduction.

Uniform random selection is a negative control for error
enrichment, not an OOD detector.

## Execution status

Read-only reproduction verified on 2026-10-18.

A new saved run has not yet been created.
