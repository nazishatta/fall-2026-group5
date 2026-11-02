# Deviations and rule changes

This file lists every time we did not follow a written rule (a protocol, a pre-registration or fixed criteria), or changed one, and what we did about it, for every week of the project. Oldest first. Nothing here has been hidden or undone.

Week 3 rows refer to `p2_som_training_visualization/som_nnsom_training/SOM_GRID_SELECTION_PROTOCOL.md`. Moved here from `docs/som/` on 2026-10-10 and given the "Week" column; the rows themselves are unchanged.

| Date | Week | Commit | What happened | Rule broken | What we did |
|---|---|---|---|---|---|
| 2026-09-15 | 3 | a4c4b0f | The grid-selection script was written to stop with an error unless 20x20 won. | The grid must be chosen by the rule, whatever the result. | Fixed in a1b8273: that script was retired and replaced by `som_grid_ablation.py`, which only reports the numbers and picks nothing. |
| 2026-09-15 | 3 | ed8c4a8 | The 10x10, 15x15 and 20x20 runs were retrained under the **same run IDs**, replacing the files behind the 20x20 choice. On the new files the rule picks 10x10. | Results must never be overwritten. | Both result tables are recorded in the protocol's "Recorded outcome". We report that the surviving 20x20 model's fingerprint doesn't match the original. |
| 2026-09-15 | 3 | (working session) | The instructor asked us to use 15x15 as an experiment. | The chosen grid should be the one used for analysis. | Recorded in "Recorded outcome". The instructor confirmed it on issue #3 (2026-10-04). |
| 2026-09-21 | 3 | 18b5ae8 | The analysis was switched to 15x15 with no written reason, the result file `docs/week_3/SOM_GRID_SELECTION_RESULT.md` was deleted, and the model fingerprint check was removed. | The outcome must be recorded, and model files must be checked. | The result table was restored into "Recorded outcome". The fingerprint check was restored in a1b8273; it now reads the expected value from the run manifest. |
| 2026-09-27 | 3 | b908f31 | Only 15x15 was re-tuned (neighbourhood 3 to 11, epochs 100 to 250), and its metrics file was overwritten. | All grids get equal treatment; no overwriting. | We ran a fair, pre-registered test where every grid got the same tuning (abef97d, cb48b99). It chose the same setting (11) for 15x15. The old file is kept; new runs use new IDs (`_v2`). |
| 2026-10-06 | 3 | abef97d, cb48b99 | Added the pre-registered comparison (rules committed before results). | Not a deviation; this is extra analysis. | Result: inconclusive (difference +0.001, 95% range −0.100 to +0.089). See `GRID_COMPARISON_RESULTS.md`. |
| 2026-10-10 | 5 | (merge of the Week 4–5 code; see git log) | Three different error-hotspot rules were in use (top 20 by Wilson lower bound; at least 20 images and Wilson lower bound above the overall error rate; top 5% by Wilson lower bound). | Not a deviation: no written rule was broken. Two rules that were never written down are replaced by the one fixed in advance. | One rule everywhere: the one in `ERROR_PATTERN_ROBUSTNESS_CRITERIA.md`, section 3. See `p2_som_training_visualization/error_pattern_analysis/HOTSPOT_RULE.md`. The Week 5 LaTeX report (2026-10-03) used the old rule and is left unchanged. |
| 2026-10-10 | 4 | (Week 4 model-check tolerance; see git log) | The Week 4 model check required the recomputed quantization error to match the Week 3 value to an absolute 1e-12. Outside EC2 (scikit-learn 1.7.2 vs 1.9.0) it differs by about 2 parts in a billion, so Week 4 could only run on EC2. | Not a deviation: the 1e-12 value was a coding choice, not part of any rule fixed in advance. | The tolerance is now relative, one part in a million. The number of neurons used must still match exactly, and the model and embedding fingerprints are still checked. Week 4 records both QE values and their relative difference in `cluster_summary.json`. |
| 2026-10-10 | 4 | (label-shuffle comparison; see git log) | Added a label-shuffle comparison for the validation weighted purity: the class labels are shuffled across images 1,000 times (seed 20261010, both in `configs/som*.yaml`, `analysis.purity_null_*`), each image keeping its neuron, and the purity is recomputed each time. Asked for by the project rules ("use a null or control"). | Not a deviation: an addition after the fact, not part of any rule fixed in advance; no existing Week 4 result changed. | Result on the 15x15 SOM: observed 0.9819 vs shuffled mean 0.1738 (95% range 0.1703–0.1774, maximum 0.1807), p = 0.001 (the smallest possible with 1,000 shuffles). Stored in `cluster_summary.json` under `splits.val.purity_label_shuffle`. |

## Phase 3 — novelty threshold and OOD benchmark

### Phase 3 score-export metrics JSON formatting repair

Recorded: 2026-11-01.

The existing untracked metrics file for
`mnist_svhn_energy_knn_score_export_s42_v1` contained one complete
JSON object followed by two literal characters (`\\n`), making it
invalid as a single JSON document.

The two invalid characters were removed in place, and one normal
newline was retained. Parsed scientific values and the run ID were
verified unchanged. The configuration SHA-256 and all seven frozen
input SHA-256 hashes matched before the formatting repair.

Original file SHA-256:
`152753ce90fe4261c682bb848a01cd1e1f61829ffe71c78043a4a157f7985401`

Corrected file SHA-256:
`70a04ea4d1c20fa841a19b2b1c82cc2b24f1d37730e46b2f45292211600f912e`

This was a formatting-only modification to an existing output file,
not a new experiment or a reproduction of the scientific results.
The original bytes were not separately preserved before modification.
The recorded metrics retain their sample-alignment limitation.

### Phase 3 configuration and result run-ID traceability

Recorded: 2026-11-01.

Four Phase 3 configuration/run-ID discrepancies were reviewed.
Historical saved results exist for QE OOD evaluation, QE random
control, and QE/U-Matrix validation, but their directory names differ
from the run IDs currently written in the corresponding YAML files.
The original configuration version used for each historical result
has not been independently established by this directory check.

The proposal-gap configuration specifies
`mnist_svhn_proposal_gap_analysis_s42_v1`, but no corresponding
saved result was found in the inspected local outputs. Its required
validation BMU assignment file is unavailable locally. Phase 2
source files, results, and DVC artifacts were not modified.

Decision: preserve historical configs and outputs unchanged.
Document the differences in the Phase 3 component README.
Do not rename historical runs, claim missing results, or overwrite
existing experiments. New executions require new run IDs.


Recorded: 2026-10-19.

### Training-calibrated QE threshold versus validation selection

The original Phase 3 proposal specified a novelty threshold at the
95th percentile of MNIST training quantization errors. The completed
SVHN experiment followed this specification. Its frozen threshold
is 1.5690279077294162.

Subsequent instructor reviews requested validation-based threshold
selection. The two protocols are not equivalent.

Decision: preserve the original training-calibrated experiment and its
recorded results. Do not retroactively replace the threshold or change
the historical run. The training-calibrated threshold is not described
as validation-selected.

A separate validation-selected threshold experiment would require a
new run ID, its selection rule fixed before evaluation, and an
independent held-out evaluation. No such result is claimed here.

### OOD dataset scope

The original Phase 3 proposal specified SVHN as the external OOD dataset.
Subsequent instructor-review feedback was interpreted as requiring
Fashion-MNIST during the current MNIST phase. Additional exploratory
Fashion-MNIST code and results were produced in response.

On 2026-10-19, Professor Jafari clarified through the teammate review
that the current scope is MNIST versus SVHN only. Fashion-MNIST and
KMNIST comparisons belong to later dataset work, not this PR.

Decision: retain the original MNIST-SVHN evaluation and frozen
training-calibrated QE threshold. Remove Fashion-MNIST-specific
code, tests, configurations, results, and current-result claims
from PR #17 through ordinary commits without rewriting Git history.
Preserve the exploratory Fashion-MNIST work privately for possible
later reuse; do not create or push a separate GitHub branch now.
No official MNIST final-test data is used for this Phase 3 tuning.

### Historical SOM configuration

Phase 3 uses the frozen 15x15 SOM employed by the previous analysis.
This does not retroactively change the earlier SOM grid-selection
results or the grid-selection protocol documented above.

### Baseline comparability

The historical One-Class SVM fit uses a seeded sample of 10,000 MNIST
training embeddings, while other fitted baselines may use 49,000.
These historical results are not treated as equal-training-budget
comparisons without qualification.

The historical Isolation Forest and Autoencoder AUROC concerns require
an explicit score-direction and reproducibility investigation. Values
must not be reversed or reselected after examining OOD outcomes merely
to improve reported performance.

### Autoencoder input-standardization diagnostic

Recorded: 2026-10-20.

The historical feature-reconstruction autoencoder was trained on raw
84-dimensional CNN embeddings and obtained AUROC 0.007719 on MNIST
validation versus SVHN test data. This unexpectedly low performance
motivated an exploratory input-standardization diagnostic.

A separate experiment fitted StandardScaler on the 49,000 MNIST
training embeddings only. The original autoencoder architecture
(84-32-16-32-84), seed 42, 20 epochs, batch size 256, and learning
rate 0.001 were retained.

New run ID: `mnist_svhn_ae_train_standardized_s42_v1`.

Measured standardized-autoencoder results:
- AUROC: 0.760492 (95% bootstrap CI: 0.753684-0.766459)
- AUPR: 0.807320 (95% bootstrap CI: 0.802132-0.812269)
- FPR at 95% TPR: 0.597810
  (95% bootstrap CI: 0.584664-0.610669)
- Evaluation: 10,500 MNIST validation and 26,032 SVHN test samples.
- Bootstrap replicates: 2,000.

The frozen CNN checkpoint, SOM, and training-calibrated QE threshold
were not changed. The scaler and autoencoder were fitted on MNIST
training features only. No official MNIST final-test data was used.

This is an exploratory follow-up designed after observing the historical
OOD results, not an independently selected final benchmark. The original
autoencoder result remains preserved and must not be overwritten or
silently replaced. The improvement is consistent with feature scaling
affecting reconstruction-based novelty scoring, but does not establish
that scaling was the only cause of the historical result.

Metrics and SHA-256 provenance are recorded in:
`outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/mnist_svhn_ae_train_standardized_s42_v1/standardized_autoencoder_metrics.json`.

### Leakage control

Training embeddings are used for fitting baseline models. MNIST
validation serves as in-distribution evaluation data. Historical
training-calibrated QE thresholding remains distinct from any future
validation-based threshold selection. The official MNIST final-test set
must remain unused until the protocol and success criteria are frozen.


### Phase 3 configuration-to-result traceability

Recorded: 2026-11-01.

The Phase 3 documentation now provides a configuration-to-result
inventory mapping the identified YAML configurations to their
configured run IDs, step folders, and observed historical results.

Three historical run-ID mismatches remain:

- QE OOD: configured `mnist_svhn_qe_dev_initial`;
  historical directory `mnist_svhn_qe_frozen_s42_initial`.
- QE random control: configured
  `qe_random_control_reproduction_initial`;
  historical directory `qe_random_control_evaluation_initial`.
- QE/U-Matrix reproduction: configured
  `qe_u_matrix_validation_reproduction_initial`;
  historical directory `qe_u_matrix_validation_s42_initial`.

The proposal-gap analysis was subsequently generated under
`u_matrix_analysis/mnist_svhn_proposal_gap_analysis_s42_v1/`.
Its analysis JSON is staged for Git tracking. Its PDF is
DVC-tracked and its remote object has been verified.

Decision: preserve all historical configurations, run IDs,
output paths, hashes, and experiment results. Do not claim
that a current configuration produced a historical result
unless its provenance is independently established.



<!-- PHASE3_RUN_REGISTRY_DEVIATIONS -->

## Phase 3 configuration-to-run traceability

An audit of the Phase 3 YAML configurations and existing output
directories identified three historical configuration-to-folder
mismatches.

1. `qe_ood_evaluation.yaml` specifies
   `mnist_svhn_qe_dev_initial`, whereas the historical folder identified
   is `mnist_svhn_qe_frozen_s42_initial`.

2. `qe_random_control.yaml` specifies
   `qe_random_control_reproduction_initial`, whereas the historical
   folder identified is `qe_random_control_evaluation_initial`.

3. `qe_u_matrix_reproduction.yaml` specifies
   `qe_u_matrix_validation_reproduction_initial`, whereas the historical
   folder identified is `qe_u_matrix_validation_s42_initial`.

The proposal gap-analysis configuration,
`phase3_proposal_gap_analysis.yaml`, now has an identified output
directory under `u_matrix_analysis`:
`mnist_svhn_proposal_gap_analysis_s42_v1`.

These discrepancies are documented rather than silently corrected.
Historical configurations and output paths are immutable for this
reorganization. Their hashes and scientific provenance must be preserved.

A configuration-to-result registry has been added to the Phase 3 README.
It distinguishes configured historical IDs, identified historical
result folders, support configurations, and future reorganization reruns.

New reruns must receive unique IDs after preflight checks. Their results
must be recorded separately and must not replace historical evidence.
