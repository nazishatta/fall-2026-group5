# Deviations and rule changes

This file lists every time we did not follow a written rule (a protocol, a pre-registration or fixed criteria), or changed one, and what we did about it, for every step of the project. Oldest first. Nothing here has been hidden or undone.

SOM-training rows refer to `p2_som_training_visualization/som_nnsom_training/SOM_GRID_SELECTION_PROTOCOL.md`. Moved here from `docs/som/` on 2026-10-10 and given the "Step" column (first written as the proposal's week number, reworded on 2026-11-17); the rows themselves are unchanged.

| Date | Step | Commit | What happened | Rule broken | What we did |
|---|---|---|---|---|---|
| 2026-09-15 | SOM training | a4c4b0f | The grid-selection script was written to stop with an error unless 20x20 won. | The grid must be chosen by the rule, whatever the result. | Fixed in a1b8273: that script was retired and replaced by `som_grid_ablation.py`, which only reports the numbers and picks nothing. |
| 2026-09-15 | SOM training | ed8c4a8 | The 10x10, 15x15 and 20x20 runs were retrained under the **same run IDs**, replacing the files behind the 20x20 choice. On the new files the rule picks 10x10. | Results must never be overwritten. | Both result tables are recorded in the protocol's "Recorded outcome". We report that the surviving 20x20 model's fingerprint doesn't match the original. |
| 2026-09-15 | SOM training | (working session) | The instructor asked us to use 15x15 as an experiment. | The chosen grid should be the one used for analysis. | Recorded in "Recorded outcome". The instructor confirmed it on issue #3 (2026-10-04). |
| 2026-09-21 | SOM training | 18b5ae8 | The analysis was switched to 15x15 with no written reason, the result file `docs/week_3/SOM_GRID_SELECTION_RESULT.md` was deleted, and the model fingerprint check was removed. | The outcome must be recorded, and model files must be checked. | The result table was restored into "Recorded outcome". The fingerprint check was restored in a1b8273; it now reads the expected value from the run manifest. |
| 2026-09-27 | SOM training | b908f31 | Only 15x15 was re-tuned (neighbourhood 3 to 11, epochs 100 to 250), and its metrics file was overwritten. | All grids get equal treatment; no overwriting. | We ran a fair, pre-registered test where every grid got the same tuning (abef97d, cb48b99). It chose the same setting (11) for 15x15. The old file is kept; new runs use new IDs (`_v2`). |
| 2026-10-06 | SOM training | abef97d, cb48b99 | Added the pre-registered comparison (rules committed before results). | Not a deviation; this is extra analysis. | Result: inconclusive (difference +0.001, 95% range −0.100 to +0.089). See `GRID_COMPARISON_RESULTS.md`. |
| 2026-10-10 | Error pattern analysis | (merge of the cluster and error pattern analysis code; see git log) | Three different error-hotspot rules were in use (top 20 by Wilson lower bound; at least 20 images and Wilson lower bound above the overall error rate; top 5% by Wilson lower bound). | Not a deviation: no written rule was broken. Two rules that were never written down are replaced by the one fixed in advance. | One rule everywhere: the one in `ERROR_PATTERN_ROBUSTNESS_CRITERIA.md`, section 3. See `p2_som_training_visualization/error_pattern_analysis/HOTSPOT_RULE.md`. The error pattern analysis LaTeX report (2026-10-03) used the old rule and is left unchanged. |
| 2026-10-10 | Cluster analysis | (Cluster analysis model-check tolerance; see git log) | The cluster analysis model check required the recomputed quantization error to match the SOM training value to an absolute 1e-12. Outside EC2 (scikit-learn 1.7.2 vs 1.9.0) it differs by about 2 parts in a billion, so cluster analysis could only run on EC2. | Not a deviation: the 1e-12 value was a coding choice, not part of any rule fixed in advance. | The tolerance is now relative, one part in a million. The number of neurons used must still match exactly, and the model and embedding fingerprints are still checked. Cluster analysis records both QE values and their relative difference in `cluster_summary.json`. |
| 2026-10-10 | Cluster analysis | (label-shuffle comparison; see git log) | Added a label-shuffle comparison for the validation weighted purity: the class labels are shuffled across images 1,000 times (seed 20261010, both in `configs/som*.yaml`, `analysis.purity_null_*`), each image keeping its neuron, and the purity is recomputed each time. Asked for by the project rules ("use a null or control"). | Not a deviation: an addition after the fact, not part of any rule fixed in advance; no existing cluster analysis result changed. | Result on the 15x15 SOM: observed 0.9819 vs shuffled mean 0.1738 (95% range 0.1703–0.1774, maximum 0.1807), p = 0.001 (the smallest possible with 1,000 shuffles). Stored in `cluster_summary.json` under `splits.val.purity_label_shuffle`. |
| 2026-10-19 | Cluster analysis | 5c55ef3 | The proposal says to assign train and test samples to their best matching neurons. We assign train and validation samples only; the cluster and error analyses never use the test split (`test_used: false` in `cluster_summary.json` and `error_summary.json`). | Not a deviation from a rule fixed in advance. It departs from the proposal wording, to follow the project rule that the test split is not used before the success criteria and selection protocol are fixed. | Kept train and validation. Test images will be assigned to neurons once, in the final evaluation, after the criteria are written down. |
| 2026-11-02 | Phase 4 | 988a59a | The proposal asks for "aggressive" augmentation of high-error SOM regions. We use mild, label-preserving changes (rotation up to 10°, shift up to 2 px, zoom 0.9–1.1, shear up to 8°, no flips, 3 copies per image). | Departs from the proposal wording; not a rule fixed in advance. | Strong changes can turn a digit into a different one (labels no longer true). The ranges are checked on validation images before any arm is trained (`--stage check`). Written in `RETRAINING_PROTOCOL.md` section 4. |
| 2026-11-02 | Phase 4 | 988a59a | The proposal builds the hard pool from "SOM cluster-boundary and error-hotspot samples". We use hotspot neurons, neurons with training purity below 0.95, and images far from their neuron (above the 95th percentile of the distance). Boundary samples are approximated by these two signals, not located exactly. | Departs from the proposal wording; not a rule fixed in advance. | Pool rule and cut-offs fixed in `RETRAINING_PROTOCOL.md` section 3 before any retraining result; the purity cut-off is a convention and not tuned. |
| 2026-11-02 | Phase 4 | 988a59a | With 20 epochs, the proposal's pacing (10% of the hard images, +10% every 5 epochs) ends at 40% of the pool, not 100%. | Not a rule broken: a consequence of the proposal's numbers. | Kept as written. A longer schedule can be added in `retraining.yaml` as a new named schedule before the protocol is frozen. |
| 2026-11-02 | Phase 4 | 988a59a | Each experiment (curriculum, augmentation) trains its own `standard` arm, so `standard` is trained twice (5 seeds each). | Not a rule broken. | Keeps each comparison self-contained and run-ID separate; the two standard arms are expected to agree up to training noise. |
| 2026-11-02 | Phase 4 | 988a59a | Retraining uses a custom in-memory training loop (same recipe as Phase 1) because the curriculum changes the training set between epochs. | Not a rule broken. | Loop is in `utils/cnn_retraining.py`; Phase 1 recipe (Adam 0.001, weight decay 1e-4, StepLR 10/0.1, 20 epochs, batch 64, best epoch on validation) is read from the config. |
| 2026-11-02 | Phase 4 | 988a59a | The augmentation transform check failed with the first ranges (maximum shift 2 px): the Phase 1 CNN's validation accuracy fell from 0.9912 on original images to 0.9583 on altered ones (drop 0.033, limit 0.02); the shift alone cost 0.025. | Not a rule broken: the protocol (section 4) says to make the ranges smaller under a new revision before any arm is trained. | Maximum shift lowered to 1 px under revision `r2`; the check then passes (0.9912 to 0.9821, drop 0.009). The failed `r1` check folder is kept as the record. Both changes were written in `retraining.yaml` and protocol section 4 before the protocol was frozen. The 2-point limit and the 1 px value are conventions, not from research. |
| 2026-11-03 | Phase 4 | 5a31e6b | The SOM re-analysis covers the augmentation arms `standard`, `augmented_som` and `augmented_random` only, not every arm. The choice was made after the retraining test results were known (the augmentation criteria were met, the curriculum criteria were not). | Not a rule broken: the arm list is fixed in `SOM_REANALYSIS_PROTOCOL.md` before any new SOM is trained. | Disclosed in the protocol's leakage control. The curriculum arms can be added in `configs/som_reanalysis.yaml` as a secondary analysis under a new run revision. |
| 2026-11-03 | Phase 4 | 5a31e6b | The re-analysis compares maps with 5 CNN seeds and a single SOM seed (42), as in Phase 2. | Not a rule broken (the proposal asks for at least 5 seeds). | Stated as a limitation in the protocol: with 5 seeds only large effects can be detected. |
| 2026-11-15 | Proposal | 4ec7679 | `Proposal.md` said its `[TBD]` thresholds and the RQ3–RQ6 owners would be filled in by the 2026-10-06 session. They stayed `[TBD]` until 2026-11-15. | Our own deadline in `Proposal.md` §6 and §7. | The values had already been fixed elsewhere before any test-set use: RQ2 in `ERROR_PATTERN_ROBUSTNESS_CRITERIA.md` (written 2026-10-06, after the first 15x15 validation results but before the robustness results; RQ2 never uses the test split), and the RQ5 macro-F1 guard (0.00157) in `RETRAINING_PROTOCOL.md`, stamped 2026-11-03 before the Phase 4 test runs. They were copied into `Proposal.md` unchanged. The RQ2 rule replaces the proposal's "x% of errors" with a held-out ratio test, so no percentage was ever chosen. Owners were filled in from who did the work. |
| 2026-11-15 | Phase 2 | 4a3d6f6 | The 10-06 session asked for the SOM grids to be tuned with Optuna, including the learning rate (alpha). Until now each grid was tuned by a 3-point neighbourhood search instead. | Not a rule fixed in advance; a session request left open for 6 weeks. | Added equal-budget Optuna tuning for all three grids (20 trials each, TPE seed 42, neighbourhood searched as a fraction of the side length, validation features only), followed by a check of the chosen setting against the 3-point choice on 5 new SOM seeds, with the protocol `p2_som_training_visualization/som_nnsom_training/OPTUNA_TUNING_PROTOCOL.md` written before any real run. NNSOM's batch SOM has no learning rate, so `init_neighborhood` and `epochs` are tuned instead. The headline model stays `som_15x15_nb11_ep250_s42_v2`; the Optuna result is reported as a check on the 3-point search. |
| 2026-11-17 | Phase 2 | 7c78e21 | Optuna tuning finished (protocol `OPTUNA_TUNING_PROTOCOL.md`, written before the runs). | Not a deviation: the rules and their use were fixed in advance. | 5-seed check, Optuna choice against the 3-point choice: 15x15 (headline) confirmed, no clear difference; 20x20 3-point setting better (Optuna worse on TE1 and TE1+2); 10x10 Optuna setting better (nb 8, 100 epochs, better on QE, TE1 and TE1+2). Headline model unchanged. The 10x10 robustness map is reported as tuned below its best setting; the robustness check is not re-run. |
| 2026-11-17 | Phase 4 | defa188 | Added an exploratory `breakdown` stage after the single test pass (Reviews #8 and #9): test errors fixed and newly made per seed against `standard`, and test error per digit with the same paired comparisons as the test pass. | Not a deviation: no criterion, threshold, setting or run changed; nothing was retrained or re-run. The breakdown reads the saved test predictions and stops unless they reproduce `test_results.json` exactly. | Reported as exploratory in the Phase 4 report (section "Errors fixed, and error per digit"). The SOM-vs-MSP test comparison asked for in Review #8 already existed (row R1 of the criteria table). |
| 2026-11-17 | Proposal | – | Moved `reports/Project_proposal.md` and `reports/Proposal.md` into a new top-level `proposals/` folder with `git mv`, and renamed `Proposal.md` to `Group_proposal.md`. The READMEs that link them were updated. | Not a deviation: no rule, criterion or threshold changed. | The two protocols fingerprinted by their runs (`HELD_OUT_CLASS_NOVELTY_PROTOCOL.md`, `NOVELTY_AFTER_RETRAINING_PROTOCOL.md`) and the older rows of this file still name `reports/Proposal.md`; they are left as written, because editing a stamped protocol would change its SHA-256. `proposals/README.md` maps the old paths to the new ones. |
| 2026-11-21 | Phases 3 and 4 | (refactor PR) | The Phase 3 and 4 code now reads the dataset from the config, as Phases 1 and 2 already did: the dataset name from the `base.yaml` next to each config (`utils/in_distribution.py`), the Phase 3 input manifest from the config's own folder (`utils/shared_provenance.py`), and the network's channels and image size from the config. `run_all.py` now also runs the steps that were run by hand (error-pattern baselines, detector budget, Optuna, feature-layer ablation, held-out classes, Phase 4), in proposal order, skipping steps whose results exist. | Not a deviation: no rule, threshold, setting, run or result changed. | Every MNIST name stays as it was (file names such as `qe_mnist_val_scores.npy`, keys such as `mnist_validation`, `added_per_digit`, `mnist_test_read`, and the wording in result files), so existing results still pass their SHA-256 and expected-value checks. A dry run of every existing `run_all.py` command plans the same MNIST steps as before (the new steps are reported as already finished). The macro-F1 guard, the protocol stamp and the single test pass remain manual stop points. |
| 2026-11-21 | Phase 3 | (Fashion-MNIST Phase 3 configs PR) | Three additions to the shared code: a new step, `novelty_benchmark/run_novelty_statements.py`, that scores all six novelty scores on the same images in one run and judges the group proposal's RQ3 and RQ4 (and a dataset protocol's own statements, for Fashion-MNIST O1 to O4); a `scores_subfolder` setting in `run_missed_svhn_figure.py` (default `score_arrays`, MNIST's layout); and `qe_ood` planned before `svhn_benchmark` in `run_all.py`, because the benchmark reproduces `qe_ood`'s metrics. | Not a deviation: no MNIST rule, setting, run or result changed. | `qe_ood` is optional for MNIST (it repeats a historical result), so MNIST's default `run_all.py` plans are unchanged; only `--with-reproductions` lists it earlier. |
| 2026-11-21 | Phase 3 | (Fashion-MNIST Phase 3 configs PR) | MNIST gets a config for the new step, `configs/novelty_statements.yaml` (run `mnist_svhn_novelty_statements_s42_v1`). Two checks the rules ask for were missing: the project rules require comparing against max-softmax, energy **and Mahalanobis** on the same embeddings, but no MNIST step gave a paired QE-against-Mahalanobis interval; and RQ4 is defined against the **best** baseline, while MNIST's check used max-softmax (energy has the higher AUROC). | Post hoc: run after MNIST's Phase 3 results were known. Not a rule change: RQ3, RQ4 and the margin are applied exactly as fixed on 2026-11-10, with the same bootstrap settings as the other Phase 3 steps. | Reported as a completion of the RQ3/RQ4 check, not as a new finding; if its verdict differs from the one in `Group_proposal.md`, both are reported and the difference is explained here. Nothing was chosen from results. |

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

**2026-11-17: near novelty within MNIST.** Instructor Review #5 (2026-10-19) also asked for "one
near-OOD pair (e.g. MNIST vs held-out MNIST classes)"; later reviews did not repeat or cancel it. The
held-out-class step (`docs/p3_novelty_detection/held_out_class_novelty/HELD_OUT_CLASS_NOVELTY_PROTOCOL.md`)
answers it without a new dataset: a LeNet-5 and a SOM are trained on 6 MNIST digits and the other 4 are
the novel images, over 5 class splits. This stays inside the 2026-10-19 scope (MNIST plus SVHN). Its
protocol, splits and statements were committed before any network or map of the step was trained. It
trains new networks, so its results do not change any earlier Phase 3 result.

**2026-11-17: two secondary analyses, added after the results were seen.** (1) The SOM re-analysis is
repeated with the least-confident (max-softmax) augmentation arm, which the primary re-analysis protocol
had left out (`docs/p4_guided_retraining/som_reanalysis/MSP_ARM_ADDENDUM.md`, new run ID; the primary run is
unchanged). (2) A same-budget comparison of the error detectors (`error_detector_budget.yaml`, change log in
`docs/p2_som_training_visualization/error_pattern_analysis/baselines/PLAN.md`). Both are labelled secondary
or exploratory and do not replace any earlier result. Outcome of (1): the least-confident arm's new maps
show no clear hotspot drop against normal training (S1 not met), and SOM guidance is not clearly better than
confidence guidance (S2, S3 not met); the shared cells reproduce the primary run exactly
(`MSP_ARM_ADDENDUM.md`, section 7).

**2026-11-17: feature-layer ablation added (Phase 2).** The proposal's paper outline asks for an ablation of
the feature layer; until now only `fc2` was used. Protocol `p2_som_training_visualization/som_nnsom_training/
FEATURE_LAYER_ABLATION_PROTOCOL.md`, written before any `fc1` embedding was extracted. It compares SOMs on
`fc1` and `fc2` of the frozen Phase 1 network with the same recipe and 5 new SOM seeds. Not a deviation: the
headline layer stays `fc2` whatever the result, and no earlier result is re-run or replaced. The SOM recipe was
tuned on `fc2`, which may favour `fc2`; this is declared in the protocol. Outcome: `fc2` is better at
locating mistakes (L1) and purer (L2); `fc1` is slightly better on SVHN novelty (L3, +0.005 AUROC).

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

Result (run 2026-11-15): QE reproduced 0.9811068. The QE-length Spearman correlation is
-0.83 pooled, -0.66 within SVHN and 0.26 within MNIST validation. Only 8 SVHN images fall in
the training-length band (against 9,447 MNIST validation images); on them QE's AUROC is 0.72
(95% interval 0.58 to 0.84). With so few SVHN images this is very uncertain, but it suggests
that much of QE's separation of SVHN comes from embedding length.

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
- The `[TBD]` thresholds in `reports/Proposal.md` section 6 were due 2026-10-06. The
  RQ4 margin was fixed on 2026-11-10 (see "Proposal thresholds and owners"); the
  others will be filled, with a written reason, before the test split is used for them.
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

### Four report figures redrawn under new run IDs

Recorded: 2026-11-15.

The PDFs written by four runs were not kept (run PDFs are not stored in git), so the
report copies could not be checked against them. Each figure was drawn again from the
same saved inputs under a new run ID; the original folders were not changed.

| Figure | Original run | New run | Config |
|---|---|---|---|
| U-Matrix on the SOM grid | `mnist_umatrix_hex_grid_s42_v1` | `mnist_umatrix_hex_grid_s42_v2` | `u_matrix_grid_figure.yaml` |
| QE versus U-Matrix on validation | `mnist_qe_umatrix_validation_figure_s42_v1` | `mnist_qe_umatrix_validation_figure_s42_v2` | `qe_umatrix_validation_figure.yaml` |
| SVHN images QE misses | `mnist_svhn_missed_qe_figure_s42_v1` | `mnist_svhn_missed_qe_figure_s42_v2` | `missed_svhn_figure.yaml` |
| Training and validation QE distributions | `qe_training_calibrated_initial` | `qe_training_calibrated_v2` | `qe_score_distribution_figure.yaml` |

- Each new run's numbers equal the original's; only the run ID and the config and PDF
  fingerprints differ. The three `_v1` figure folders held only that metrics file, without
  their PDF, so they were removed as duplicates (git history keeps them).
  `qe_training_calibrated_initial` is kept, because later steps read it.
- The new runs' PDFs are tracked with DVC, so they cannot be lost the same way.
- `qe_novelty.yaml` still names `qe_training_calibrated_initial`, because later steps
  read that run's `qe_summary.json` through it; the figure run has its own config.
- The report copies are taken by hand from these runs' `pdf/` folders; each caption
  names its source run.
