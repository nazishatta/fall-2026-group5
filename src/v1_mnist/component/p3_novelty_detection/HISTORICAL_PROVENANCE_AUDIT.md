# Phase 3 Historical Provenance Reconciliation

## Scope

This audit covers three historical Phase 3 experiments.

It does not modify historical configurations, experiment
outputs, numerical results, or run identifiers.

## 1. QE OOD Evaluation

Historical run ID:
`mnist_svhn_qe_frozen_s42_initial`

Current configured run ID:
`mnist_svhn_qe_dev_initial`

Historical result:
`qe_ood_metrics.json`

Available evidence:
- SOM SHA-256 recorded.
- CNN SHA-256 recorded.
- MNIST training-feature SHA-256 recorded.
- Training and validation QE SHA-256 values recorded.
- Training-derived QE threshold: 1.5690279077294162.

Assessment:
The experiment has substantial recorded artifact provenance.
However, its exact original YAML revision has not been
identified in the inspected Git history.

Status: PARTIALLY VERIFIED.

## 2. QE Random Control

Historical run ID:
`qe_random_control_evaluation_initial`

Current configured run ID:
`qe_random_control_reproduction_initial`

Historical result:
`qe_random_control_metrics.json`

Available evidence:
- Source run: `qe_training_calibrated_initial`.
- Random seeds: 11, 23, 37, 47, 59.
- Threshold source: training only.

Assessment:
The source run and random seeds are recorded.
The exact generating YAML revision remains unidentified.

Status: PARTIALLY VERIFIED.

## 3. QE and U-Matrix Validation

Historical run ID:
`qe_u_matrix_validation_s42_initial`

Current configured run ID:
`qe_u_matrix_validation_reproduction_initial`

Historical result:
`qe_u_matrix_validation_initial.json`

Available evidence:
- Dataset: MNIST.
- QE threshold: 1.5690279077294162.

Assessment:
The threshold is consistent with the recorded historical
QE OOD threshold, but this agreement alone does not prove
identical model artifacts or generating configurations.

Status: LIMITED PROVENANCE.


## Verified QE OOD Artifact Hashes

All five SHA-256 values recorded in the historical
`mnist_svhn_qe_frozen_s42_initial` metrics were independently
matched to existing EC2 artifacts by hashing file contents.

| Artifact | Verification |
|---|---|
| CNN checkpoint | Exact SHA-256 match |
| MNIST training embeddings | Exact SHA-256 match |
| Training QE distances | Exact SHA-256 match |
| Validation QE distances | Exact SHA-256 match |
| SOM model | Exact SHA-256 match |

Verified SOM model:

`outputs/v1_mnist/p2_som_training_visualization/som_nnsom_training/som_models/som_15x15_nb11_ep250_s42_v2`

An additional SOM file named `som_15x15_seed42_v1`
has identical file contents and the same SHA-256.

The matching artifacts establish the integrity and
availability of the recorded historical inputs.

They do not independently establish which exact YAML
revision generated the historical experiment.

### Verified artifact locations and SHA-256 values

The following recorded historical hashes were matched against
the existing EC2 files:

| Artifact | SHA-256 |
|---|---|
| SOM | `e02d4849353348b7b7a0d7dca61a17d34a80b20d9d476d69542f1456555c1831` |
| CNN | `0d5f3d6543de6e87fb6510c4790ca131b996a6f86d0f96880197858d8a150b6d` |
| Training embeddings | `26faeeac9af896cbd72776b0ad87bc3573218916894d615cedcbdb3c9cd68d65` |
| Training QE | `10f86df34b5677776e202dde1e59b49d5df0d8126ea3acca28043748908a238d` |
| Validation QE | `b0983bc8dba74998195d88011eabd1431bec449958932203c2beb3d4ec136186` |

Verified repository-relative files:

- SOM: `outputs/v1_mnist/p2_som_training_visualization/som_nnsom_training/som_models/som_15x15_nb11_ep250_s42_v2`
- CNN: `outputs/v1_mnist/p1_model_training_baseline/cnn_baseline_training/checkpoints/lenet5_best.pth`
- Training embeddings: `outputs/v1_mnist/p1_model_training_baseline/cnn_baseline_training/embeddings/train_features.npy`
- Training QE: `outputs/v1_mnist/p2_som_training_visualization/cluster_analysis/som_15x15_nb11_ep250_s42_v2/assignments/train_bmu_distance.npy`
- Validation QE: `outputs/v1_mnist/p2_som_training_visualization/cluster_analysis/som_15x15_nb11_ep250_s42_v2/assignments/val_bmu_distance.npy`

These are file-content matches, not proof of the exact historical
executing configuration.

## Git History Investigation

A read-only Git history search examined the corresponding
historical configuration-file revisions.

- QE OOD: 5 revisions inspected; no exact run-ID match.
- Random control: 1 revision inspected; no exact run-ID match.
- QE + U-Matrix: 1 revision inspected; no exact run-ID match.

These searches are limited to the inspected file histories.
They do not establish that no original configuration exists
in another path, uncommitted workspace, or external archive.

## Leakage Control

No new experiment was executed for this audit.

No MNIST final-test data was accessed.

No hyperparameters, thresholds, or selection criteria
were changed based on SVHN evaluation outcomes.

## Conclusion

The three historical run-ID discrepancies are documented.

The original configuration-to-result mappings remain
unverified.

Historical results must not be presented as fully
reproducible from the current YAML files.

Any future reproduction must use an independently
identified configuration and a new run ID.

No historical output may be overwritten.
