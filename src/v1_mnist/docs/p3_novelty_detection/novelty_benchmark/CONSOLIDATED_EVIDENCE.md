# Phase 3 consolidated benchmark provenance

The frozen Phase 3 evaluation uses 10,500 MNIST validation samples and 26,032 SVHN test samples. This document identifies the authoritative source files so that reproduced or copied metrics do not appear to be independent experimental evidence.

**Headline MSP, energy, Mahalanobis and kNN source:** `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/mnist_svhn_required_baselines_knn_s42_v1/required_baseline_metrics.json`.

**Historical QE, U-Matrix, Isolation Forest, One-Class SVM and unscaled-autoencoder source:** `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/mnist_svhn_five_method_comparison_s42_initial/comparison_metrics.json`.

**Paired QE–MSP evidence:** `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/mnist_svhn_qe_msp_reproduction_s42_v1/qe_msp_paired_metrics.json`.

The consolidated CSV is a **descriptive joining of saved runs**, not a claim that U-Matrix and all new baselines were executed in a single identical-fit run. The historical SVM used fewer training examples than some other methods; the separate five-seed fair-budget run is exploratory and should be presented independently. No historical results are overwritten.

Historical JSON files containing `[TBD]` are **validation-only development-stage records**. Retain them unchanged for provenance. Later SVHN evaluations are recorded under distinct run IDs. Old documentation referring to 73/168/172 tests records earlier environments and commits, not the latest passing test count. Always cite each test run with the exact branch SHA, environment and command; do not replace all historical counts with a new figure.

The random-score null simulation interval is not a detector's bootstrap confidence interval. The additional standardized-autoencoder run is post-hoc exploratory and does not supersede the original unscaled autoencoder experiment. The official MNIST final-test set remains untouched until selection criteria are frozen and a guarded evaluation is implemented.

## Headline SVHN results (MNIST validation vs SVHN test)

Frozen LeNet-5 (84-D `fc2` embeddings), 15x15 SOM, MNIST validation 10,500 (not
novel) vs SVHN test 26,032 (novel); OOD is the positive class and a higher score
means more novel. Values are from the source files named above and from the QE
reproduction in `HISTORICAL_PROVENANCE_AUDIT.md`. 95% bootstrap intervals for AUROC.

| Method | AUROC | AUPR | FPR at 95% TPR |
|---|---:|---:|---:|
| Energy | 0.9998 [0.9997, 0.9999] | 0.9999 | 0.0004 |
| Max-softmax | 0.9985 [0.9982, 0.9987] | 0.9994 | 0.0078 |
| QE (95th-percentile training cutoff) | 0.9811 [0.9790, 0.9831] | 0.9766 | 0.0453 |
| kNN (k=5) | 0.9766 [0.9743, 0.9787] | 0.9739 | 0.0592 |
| Mahalanobis | 0.7764 [0.7688, 0.7832] | 0.7859 | 0.3131 |

- **FPR at 95% TPR** is the lowest false-positive rate among ROC points with TPR of
  at least 0.95. It is a proportion, and it is **not** the frozen QE cutoff's false
  alarm rate (5.1%). Do not mix it with records that define FPR differently.
- **Paired QE vs max-softmax** (stratified paired bootstrap, 2,000 iterations,
  seed 42): QE minus max-softmax AUROC is -0.0174, 95% interval [-0.0194, -0.0154].
  Max-softmax is better on this benchmark. The other pairs were not tested.
- **Limits:** one frozen CNN, one external dataset, one seed. Energy temperature is
  fixed at 1.0 and Mahalanobis uses a shared Ledoit-Wolf covariance. These settings
  were specified after the earlier results and are not claimed as preregistered.
  Pairing assumes the identical deterministic SVHN ordering (the historical QE
  arrays have no per-image IDs).

## Pipeline scope

The Phase 3 step runners in `pipeline/p3_novelty_detection/` list every step with its own config in a dry run (checked by `tests/pipeline/test_phase3_runners.py`). Shared `pipeline/common.py` and `pipeline/run_all.py` require teammate agreement; a dedicated follow-up PR was previously agreed for those shared files. Do not imply that the Phase 3 runners mean shared end-to-end pipeline integration.

## Multi-seed fitting-budget diagnostic

The exploratory fair-budget configuration
`fair_budget_baselines.yaml` uses `seed_list` rather than `seeds`.
The runner explicitly reads `config["seed_list"]`, requires at least
five distinct seeds, and records them under `seeds` in its results.

The completed experiment used seeds 42, 43, 44, 45 and 46, with
10,000 MNIST training embeddings per method and identical selected
training indices within each seed. Isolation Forest, One-Class SVM
and autoencoder AUROC means, sample standard deviations and
per-seed values are saved in
`mnist_svhn_fair_budget_baselines_s42_46_v1/fair_budget_metrics.json`.

The experiment equalizes training-example counts, not necessarily
computation time or optimization budgets. It is exploratory and
does not replace the historical benchmark.

The original YAML field must remain unchanged because the recorded
configuration SHA-256 is part of the experiment provenance.

## Frozen configuration provenance

The canonical Phase 3 input fingerprints are maintained in
`src/v1_mnist/component/configs/phase3_shared_provenance.yaml`.
The shared verifier checks the frozen CNN checkpoint, MNIST training
and validation embeddings, and the canonical SVHN test-file checksum
for active runners that supply these inputs.

Historical experiment configurations are intentionally preserved.
Saved result JSONs include their configuration SHA-256 values, and
the original configuration files match those recorded fingerprints.

Verified preserved configurations:

- `autoencoder_standardized_diagnostic.yaml`:
  `c74e98b1bde72782fec4f921adb21965da759dcff9a00a94bff9f3f8f8ee7fec`
- `fair_budget_baselines.yaml`:
  `2d684e215472d30225b41db972783deb92cfc578f98202398278c8622990a043`
- `svhn_knn_baselines.yaml`:
  `d34d49d02f4223373177e51d9dbd1e3d7ea372796947ca861180322587fd7cc2`

Removing duplicate fingerprints directly from these original
configurations would alter their hashes and break saved-run provenance.
Future fully deduplicated configurations require distinct config
filenames and fresh run IDs. No existing run or result is overwritten.

The historical benchmark's duplicate hash fields remain an
explicit compatibility exception, not a claim that every configuration
has been fully deduplicated.

## Baseline scores that look odd

Isolation Forest (AUROC about 0.28) and the raw autoencoder (about 0.006) score
below chance on SVHN, and the autoencoder gave very different numbers across
runs. A controlled run (`mnist_svhn_autoencoder_scaling_diagnostic_s42_46_v1`,
5 seeds, exploratory) shows that embedding length alone separates MNIST from SVHN
(length-only AUROC 0.00007; mean length 40.3 vs 5.8), that the raw autoencoder
follows it, and that standardizing the input fixes the direction (0.979 ± 0.017
at 10,000 training rows, 0.902 ± 0.072 at 49,000). Isolation Forest stays below
chance (0.280 ± 0.085 and 0.275 ± 0.043). The earlier single-run 0.760 is **not
reproduced** and its cause is **not identified**. Details, per-seed ranges and the
open questions are in `BASELINE_SCORE_ANALYSIS.md`.

## QE cutoff sensitivity (MNIST validation vs SVHN test)

Run `mnist_svhn_qe_threshold_sweep_s42_v1`. Cutoffs come from MNIST training QE
only; the 95th percentile (1.5690279077294162) stays the headline cutoff and no
percentile was chosen from SVHN results. SVHN was seen earlier, so this is a
post-hoc sensitivity analysis. Wilson 95% intervals in brackets.

| Cutoff percentile | MNIST validation false alarms | SVHN detected |
|---:|---:|---:|
| 90 | 10.4% [9.8, 11.0] | 99.2% [99.1, 99.3] |
| 95 (headline) | 5.1% [4.7, 5.6] | 96.2% [95.9, 96.4] (996 missed) |
| 97.5 | 2.5% [2.3, 2.9] | 83.8% [83.3, 84.2] |
| 99 | 0.95% [0.78, 1.16] | 18.0% [17.5, 18.5] |
| 99.5 | 0.51% [0.39, 0.67] | 0.008% (2 of 26,032) |

- **AUROC does not depend on the cutoff:** 0.981 (95% bootstrap interval 0.979 to
  0.983), AUPR 0.977. A label-shuffle control gives a mean of 0.4999 (largest of
  1,000 shuffles 0.511), so the result is far from chance.
- **The usable operating point is sensitive to the cutoff.** About 66% of SVHN
  images have a QE between the 97.5th and 99th percentile cutoffs (1.714 and
  1.894). Moving the cutoff across that band drops detection from 84% to 18%. With
  this SOM, the proposal's "99th percentile" option would catch few SVHN images
  even though AUROC is high.

## Where the QE detector misses (SVHN on the SOM grid)

Run `mnist_svhn_som_overlay_s42_v1`. Descriptive only, nothing tuned. SVHN test is
the novel set; MNIST validation and the training class table are references. The
rebuilt SVHN QE scores match the saved historical ones within 9.8e-07.

- At the 95th-percentile cutoff the detector misses 996 of 26,032 SVHN images
  (3.8%).
- SVHN images land on only 28 of the 225 neurons. 22,542 of them (86.6%) land on
  neurons whose dominant MNIST class is 4, and neuron 79 alone receives 22,521.
- The misses are concentrated: neurons 79, 80, 60 and 72 hold 841 of the 996
  (84.4%). Neuron 79 holds 355 (a 1.6% miss rate there). Neuron 60 (dominant class
  1) misses 162 of 360 images (45%).
- By dominant MNIST class of the landing neuron, the miss rate is 45.7% for
  class 1 (169 of 370), 14.8% for class 5, 13.7% for class 8, 10.4% for class 0 and
  1.7% for class 4. Classes with fewer than 30 SVHN images are too small to
  compare.
- Missed images are mostly **not** typical for their neuron: the median ratio of a
  missed image's QE to the median MNIST validation QE in the same neuron is 1.23,
  and only 7.6% of missed images are at or below that median. So they fit their
  cell somewhat worse than a typical MNIST digit, but not enough to pass the
  global cutoff.

This describes where the score fails. It does not show why, or that the cause is
visual.
