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

Status: NUMERICALLY VERIFIED; original YAML identity unverified.

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

Status: AGGREGATE STATISTICS VERIFIED; historical seeded counts and original YAML identity unverified.

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

Status: NUMERICALLY VERIFIED; original YAML identity unverified.


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


## Verified Numerical Reproduction — 2026-11-02

Read-only EC2 reproduction; historical outputs unchanged.

### QE OOD
- Historical run: `mnist_svhn_qe_frozen_s42_initial`
- Current configuration run: `mnist_svhn_qe_dev_initial`
- AUROC: `0.9811068135920626` (exact match)
- AUPR: `0.9765546499418931` (exact match)
- FPR@95TPR: `0.04533333333333334` (exact match)
- AUROC CI: `[0.9790482566328621, 0.9830887456280913]`
- All five recorded artifact SHA-256 hashes matched.

### QE random control
- Historical aggregate results reproduced exactly.
- Validation samples: `10500`; QE flags: `539`.
- CNN errors: `92`; captured errors: `22`.
- Exact one-sided p-value: `8.834306580361853e-10`.
- Historical seeded counts: `[7, 4, 5, 7, 7]`.
- Reproduced seeded counts: `[6, 5, 5, 5, 5]`.
- Original seeded sampling procedure remains unidentified.

### QE/U-Matrix
- Mean QE: `0.9634832918297878`.
- Mean U-Matrix: `0.7809652917712436`.
- Spearman correlation: `0.4396258885327658`.
- Historical maximum QE reconstruction difference:
  `4.860474741713583e-06`.
- Independently recomputed: `4.860474741713583e-06`.
- Absolute difference: `0.0`.

### Original YAML provenance
Five historical revisions of `qe_ood_evaluation.yaml` were
inspected. All used `mnist_svhn_qe_dev_initial`, not the
historical frozen run ID. The historical JSON has no
`config_sha256` field. The exact original generating YAML
identity remains unestablished.

Current YAML SHA-256:
`7ba80772bce316c84d22b216272ca0152a047ef4f2ca89ed6571540e21b43377`.

These are verified numerical reproductions, not proof that
current YAML files are byte-identical to original configurations.

## Leakage Control

Read-only numerical reproductions were executed on EC2 on 2026-11-02. No new training or saved experimental run was created.

No MNIST final-test data was accessed.

No hyperparameters, thresholds, or selection criteria
were changed based on SVHN evaluation outcomes.

## Conclusion

The three historical run-ID discrepancies are documented.

The original configuration-to-result mappings remain
unverified.

The principal historical numerical results have been reproduced using current configurations. This does not establish the identity of the original generating YAML revisions.

Future reproductions must use an explicitly documented
configuration with verified input fingerprints and a new,
non-overwriting run ID. Current reproduction configurations
are not claimed to be the original historical YAML revisions.

No historical output may be overwritten.
