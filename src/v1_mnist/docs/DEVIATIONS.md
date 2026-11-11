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
Document the differences in the [Phase 3 configuration registry](p3_novelty_detection/README.md#configuration-to-result-traceability).
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

Recorded: 2026-11-01; updated 2026-11-02.

The initial audit identified three historical configuration-to-result
run-ID mismatches. Their exact generating YAML revisions remain
unverified. The proposal-gap result, initially unavailable during
the earlier inspection, was subsequently generated under
`u_matrix_analysis/mnist_svhn_proposal_gap_analysis_s42_v1/`.

Its analysis JSON was prepared for Git tracking, and its PDF was
DVC-tracked and verified in the remote storage.

The authoritative configuration-to-result mapping is maintained in
[Phase 3 documentation](p3_novelty_detection/README.md#configuration-to-result-traceability).


Decision: preserve historical configurations, output paths, run IDs,
SHA-256 hashes, and scientific results. Directory existence alone
does not establish the original generating configuration. Future
executions require new descriptive run IDs and provenance checks.

## Phase 3 historical reorganization run naming

Recorded: 2026-11-02.

During Phase 3 directory restructuring and runner-path verification,
new experiment executions were assigned unique run identifiers beginning
with `reorg_20261101_` or `reorg_20261102_`. The names indicate the
execution occasion rather than primarily identifying the scientific
experiment. The configuration directory
`configs/phase3_reorganization/reorg_20261102/` follows the same
convention.

This naming choice does not follow the project's requirement that
runs and configurations be named by their scientific purpose.

The existing run directories and their configurations are preserved
because run identifiers, recorded paths, output files, and hashes form
part of the experiment provenance. Renaming historical results could
break traceability and reproducibility.

Going forward, new run IDs and configuration paths must describe the
experiment or analysis, include appropriate seed/version identifiers,
and pass the existing run-ID and SHA-256 preflight checks. Dates or
reorganization activity must not replace descriptive experiment names.

No historical outputs, run IDs, or configurations were modified to
resolve this naming deviation.

**Cleanup, 2026-11-03.** Two duplicate run folders were deleted. Git history keeps them.

- `mnist_svhn_qe_msp_paired_s42_v1`: every number and input hash is identical to
  `mnist_svhn_qe_msp_reproduction_s42_v1`, which stays.
- `reorg_20261102_ood_comparison_qe_s42_v1`: it only repeated QE numbers already saved
  in `mnist_svhn_qe_frozen_s42_initial`. Nothing referred to it.

**Cleanup, 2026-11-10.** Fifteen repeat run folders, the fifteen configs that wrote them and two
unused copies of `base.yaml` were deleted, without re-running anything. Git history keeps them.

- All five `reorg_20261101_` folders. The `reorg_20261102_` runs repeat the same steps with
  the same numbers, so the first repeat added nothing.
- Ten `reorg_20261102_` folders whose original is already traceable to today's code: for each,
  the original records a config SHA-256 that equals today's config file (or, for the historical
  baseline verification, today's config has the same run ID), and the repeat differs from the
  original only in run ID, config hash and figure-file hash.

Five `reorg_20261102_` folders and their five configs (in `configs/phase3_reorganization/reorg_20261102/`)
are kept. They are the only runs made by today's code for results whose original config is not
in git history (`mnist_svhn_qe_frozen_s42_initial`, `qe_random_control_evaluation_initial`,
`qe_u_matrix_validation_s42_initial`), for the QE summary those three read, and for
`mnist_svhn_five_method_comparison_s42_initial`, whose config was edited after it ran. The
random-control repeat differs from its original in six per-seed counts; the five-method repeat
differs in the U-Matrix FPR at 95% TPR (0.0701 vs 0.0703). Neither difference was hidden by
the cleanup. The kept configs were not moved or renamed, because their recorded hashes and the
paths inside them are part of the provenance.

`outputs/v1_mnist/p3_novelty_detection/run_manifest.csv` now lists every remaining run folder
with the script and config that wrote it and how that was checked.

**Further cleanup, 2026-11-10.** Done without re-running any experiment.

- `score_export_metrics_validated.json` was removed from
  `mnist_svhn_energy_knn_score_export_s42_v1`. It was a byte-identical copy of
  `score_export_metrics.json` that no script writes or reads.
- `mnist_validation_reproducibility_s42_initial` was removed. It was a hand-written record of
  an early EC2 rerun check; no script wrote it, and `mnist_historical_baseline_verification_s42_v1`
  recomputes the same baselines with code. Git history keeps both files.
- `consolidate_phase3_evidence.py` now reads `configs/phase3_evidence_consolidation.yaml`
  instead of run IDs, paths and counts written in the code. The config was written after
  `mnist_svhn_consolidated_evidence_s42_v1` ran; its values equal the sources and counts that
  run recorded, and `test_consolidation.py` checks this.
- Three SVHN scripts took the image size from a literal `(28, 28)` and the unified benchmark
  wrote out the training-label path. Both now come from the validated settings and the shared
  provenance manifest. Nine scripts share one SVHN loader and one run-ID check instead of
  their own copies. No config that produced a saved result was edited.

### Proposal thresholds and owners

Recorded: 2026-11-10.

**RQ4 margin.** "Competitive" now means within 0.01 AUROC of the best baseline: the lower
end of the 95% paired bootstrap interval for (SOM AUROC minus best-baseline AUROC) must be
above -0.01. This was fixed on 2026-11-10, after the development results (MNIST validation
versus SVHN) had been seen: QE trails energy by about 0.019 AUROC and max-softmax by 0.0174
(paired interval -0.0194 to -0.0154). The strict value was chosen so that it could not be read
as fitted to let QE pass. It is larger than the bootstrap noise (QE's 95% interval is about
+/-0.002) and equals about one more misranked pair in a hundred at AUROC near 0.98. On the
development numbers QE would not count as competitive. The margin will not change before or
after the held-out test split is used.

**Still open.** The RQ2 and RQ5 thresholds and the RQ3 to RQ6 owners in section 7 are still
`[TBD]`; the owners are not decided yet. The thresholds must be fixed, with a written reason,
before the held-out test split is used for those questions.

### QE-norm dependence check

Recorded: 2026-11-10.

SVHN embeddings are much shorter than MNIST ones, and a length-only score already separates
them almost perfectly (`mnist_svhn_autoencoder_scaling_diagnostic_s42_46_v1`). To test whether
part of QE's AUROC is a length effect, `qe_norm_dependence` (config `qe_norm_dependence.yaml`,
run ID `mnist_svhn_qe_norm_dependence_s42_v1`) reports the QE-length correlation and QE's AUROC
among images whose length lies between the 5th and 95th percentiles of MNIST training
lengths. The band and every other setting were fixed in the config before the first run. The
run stops unless QE reproduces the recorded AUROC (0.9811068). It is exploratory, because SVHN
has been examined.

### MNIST validation used in place of the held-out test split

Recorded: 2026-11-03.

**What "test split" means in this project.** The 70,000 MNIST images were pooled and
split once (stratified, seed 42) into 49,000 training, 10,500 validation and 10,500
test images. The test split is therefore the project's own held-out split. It is
**not** the official 10,000-image MNIST test set. Older code strings, config comments
and saved JSON files say "official MNIST final-test"; they mean this held-out split.
They were not edited, because changing a config changes its recorded SHA-256.

**Where the test split has been used.** Phase 1 evaluated it once, after training, to
report the CNN's accuracy (99.18%, `cnn_baseline_training/metrics/test_evaluation.json`),
and test embeddings were saved. No decision, threshold or selection used it. No Phase 2
or Phase 3 result uses it.

The proposal says to score "each test sample". All Phase 3 results so far use the
10,500 MNIST **validation** images instead. This is on purpose: Phases 3 and 4 are
still being developed, and the test split should stay a one-time final check for them.

- Results are development results, not final ones. SVHN results are post-hoc
  exploratory, because SVHN was inspected earlier.
- RQ3 and RQ4 are not called "supported" or "not supported" yet.
- The `[TBD]` thresholds in `reports/Proposal.md` section 6 (for example the RQ4
  "within `[TBD]` AUROC" margin) were due 2026-10-06 and are still empty. They
  will be filled, with a written reason, before the test split is used for them.
- For the Phase 3 and 4 research questions, the test split will be used once, at
  the end, for all of them together, under a new run ID.

### Three Phase 3 steps are skipped by `run_all.py` by default

Recorded: 2026-11-03.

The steps `qe_ood`, `qe_random_control` and `qe_u_matrix` only re-create results
that already exist under the original folders (`mnist_svhn_qe_frozen_s42_initial`,
`qe_random_control_evaluation_initial`, `qe_u_matrix_validation_s42_initial`). Their
configs use different run IDs from those folders, and the `reorg_...` runs already
reproduced the originals.

- `run_all.py` skips these three steps unless `--with-reproductions` is given.
- Later steps still read the original folders, so a from-scratch run is not fully
  reproducible from the code alone. This is a known limit.
- No original run ID, config or result was changed.

### Outdated Phase 3 working documents removed

Recorded: 2026-11-03.

`docs/p3_novelty_detection/` now holds only protocol, results and reproduction
documents, like the Phase 1 and 2 docs folders. Five working documents were
deleted because they were outdated or duplicated other records: the proposal
gap-analysis handover note, the historical development log, the Phase 3 work
report, the historical evidence status tracker and the result relocation record.
Git history keeps all of them.

- The headline SVHN comparison table moved into `CONSOLIDATED_EVIDENCE.md`.
- The facts that the original autoencoder checkpoint and package versions cannot
  be recovered moved into `HISTORICAL_PROVENANCE_AUDIT.md`.
- No output, config or run ID was changed.
