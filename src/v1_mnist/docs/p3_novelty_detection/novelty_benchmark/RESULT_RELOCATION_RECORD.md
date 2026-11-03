# Historical Phase 3 result relocation

Recorded: 2026-10-19.

These tracked JSON files were relocated using `git mv`.
Original file contents and SHA-256 hashes were preserved.

| Original path | Current path |
|---|---|
| `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/autoencoder_mnist_validation_initial.json` | `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/autoencoder_mnist_validation_s42_initial/autoencoder_mnist_validation_initial.json` |
| `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/classical_mnist_validation_initial.json` | `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/classical_mnist_validation_s42_initial/classical_mnist_validation_initial.json` |
| `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/reproducibility_verification.json` | `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/mnist_validation_reproducibility_s42_initial/reproducibility_verification.json` |
| `outputs/v1_mnist/p3_novelty_detection/u_matrix_analysis/qe_u_matrix_validation_initial.json` | `outputs/v1_mnist/p3_novelty_detection/u_matrix_analysis/qe_u_matrix_validation_s42_initial/qe_u_matrix_validation_initial.json` |

## Historical provenance

The original reproducibility JSON retains its original internal paths.
Those paths describe the locations at the time of the historical run.
This relocation record maps them to the current Git paths.
No historical measurement, run ID, or saved provenance hash was edited.

## Leakage control

This operation reorganized metadata files only.
No model training, tuning, or OOD evaluation occurred.
