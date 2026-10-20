# U-Matrix Validation Reproduction

## Verified execution — 2026-10-18

Command (read-only):

    python -m src.v1_mnist.component.pipeline.p3_novelty_detection.run_qe_u_matrix

Frozen SOM: `som_15x15_nb11_ep250_s42_v2`

SOM SHA-256:
`e02d4849353348b7b7a0d7dca61a17d34a80b20d9d476d69542f1456555c1831`

## Historical versus reproduced results

| Metric | Historical | Reproduced |
| --- | ---: | ---: |
| Validation samples | 10500 | 10500 |
| SOM neurons | 225 | 225 |
| Mean QE | 0.9634832918297878 | 0.9634832918297875 |
| Mean U-Matrix | 0.7809652917712436 | 0.7809652917712436 |
| Spearman correlation | 0.4396258885327658 | 0.4396258885327658 |

The QE-mean difference is approximately 3e-16 (floating-point rounding).

## Leakage control

The reproduction used frozen MNIST validation BMU assignments,
validation QE distances, and the trained 15x15 SOM.
No MNIST final-test samples or external OOD data were used.

## Limitations

- NNSOM 1.8.3 was used locally.
- The original SOM includes a MinMaxScaler serialized with
  scikit-learn 1.9.0; local scikit-learn 1.4.2 emitted an
  InconsistentVersionWarning during loading.
- The historical maximum QE reconstruction difference
  (4.860474741713583e-06) was not independently remeasured.
- No new run was saved; execution was read-only.
- The historical JSON was not modified.
