# Protocol: novelty detection on Fashion-MNIST (Phase 3)

Part of the Fashion-MNIST protocol, one file per phase:
[Phase 1](../p1_model_training_baseline/MODEL_TRAINING_PROTOCOL.md) ·
[Phase 2](../p2_som_training_visualization/SOM_ANALYSIS_PROTOCOL.md) ·
[Phase 3](NOVELTY_DETECTION_PROTOCOL.md) ·
[Phase 4](../p4_guided_retraining/RETRAINING_PROTOCOL.md).
Changes and deviations: [`../DEVIATIONS.md`](../DEVIATIONS.md).

Written on 2026-11-17, **before any Fashion-MNIST data was loaded, any network trained or any result seen**,
and committed as one file in `5dd164c` (`docs/FASHION_MNIST_PROTOCOL.md`). The outside set, left open in that
version, was fixed on 2026-11-21 (amendment A1 in `../DEVIATIONS.md`), before any Phase 3 run. Split into one
file per phase on the same day with no rule changed. Sections 1 to 6 are not changed; a change needs a written
reason in `../DEVIATIONS.md` and, for a run already made, a new run ID.

**Rule for every phase:** every step done for MNIST is done for Fashion-MNIST, with the MNIST design copied
unchanged, unless the dataset makes that impossible. Inputs: the Phase 1 network and embeddings and the Phase 2
headline map ([Phase 1](../p1_model_training_baseline/MODEL_TRAINING_PROTOCOL.md) and
[Phase 2](../p2_som_training_visualization/SOM_ANALYSIS_PROTOCOL.md) protocols, section 1).

## 1. In-distribution reference

The 10,500 Fashion-MNIST validation images. The Fashion-MNIST test split is not used in Phase 3.

## 2. Outside set

**Never used for training, fitting, thresholds or choices.** One outside set: **SVHN** (Street View House
Numbers; Netzer et al., 2011), the same set as for MNIST. Chosen by the professor and written here on
2026-11-21, before any Phase 3 step was run on Fashion-MNIST and before any novelty score was computed with
the Fashion-MNIST network (amendment A1). The other candidates (MNIST, KMNIST) are not used.

| Item | Fixed value |
|---|---|
| Split and size | SVHN test split, cropped-digit format (`test_32x32.mat`), all 26,032 images, in file order; the SVHN training and extra splits are not used |
| File | `data/ood_svhn/test_32x32.mat`, the same file as for MNIST, SHA-256 `cdce80dfb2a2c4c6160906d0bd7c68ec5a99d7ca4831afa54f09182025b6a75b`, checked before every run; not downloaded again |
| Conversion | the MNIST transform, unchanged (`ood_adapters.image_transform("svhn", (28, 28))`): RGB to grey with the ITU-R BT.601 weights (0.299, 0.587, 0.114), resize from 32x32 to 28x28 with bilinear interpolation and anti-aliasing, pixels scaled to [0, 1]. No normalisation, no inversion of light and dark, no cropping |
| Labels | the SVHN digit labels are not used by any score; images are labelled only as "outside" |
| Licence | the SVHN authors allow non-commercial use only; this study is non-commercial research |

Fashion-MNIST images are divided by 255 in the same way (Phase 1 protocol, section 1), so both sets reach the
network on the same [0, 1] scale and at the same size.

*Why this set is a sound test, and what it can and cannot show.* SVHN and Fashion-MNIST share no class
(house-number digits against clothing), so every SVHN image is a true outside image. SVHN also differs in
how the images look (photographs with textured, often bright backgrounds, against centred objects on a black
background), so it tests a large shift: far novelty. Using the same SVHN images as for MNIST keeps the
outside set fixed while the training data changes, so the MNIST and Fashion-MNIST AUROCs can be set side by
side as the same test on two networks. They are not pooled, and no statistical test is made between the two
datasets (they are separate networks and maps). Near novelty, where outside images resemble the training
classes, is tested separately by the held-out classes (section 4). A set that is visually close to
Fashion-MNIST, such as MNIST, is a known hard case for some novelty scores (Nalisnick et al., 2019 report
it for likelihood models trained on Fashion-MNIST); it is not tested here (section 6).

*Imbalance.* 26,032 outside images against 10,500 validation images (71.3% outside). AUROC and FPR at 95% TPR
do not depend on this ratio; AUPR-out does, so it is reported with its chance level of 0.713.

## 3. Scores, measures and statements

**Scores** (higher means "more novel"; anything fitted is fitted on the Fashion-MNIST training embeddings
only; the functions are the MNIST Phase 3 functions, called unchanged):

| ID | Score |
|---|---|
| S1 | SOM quantization error (QE) |
| S2 | U-Matrix value of the best-matching neuron |
| S3 | max-softmax: 1 minus the largest softmax probability |
| S4 | energy: minus log-sum-exp of the logits, temperature 1 |
| S5 | Mahalanobis: smallest squared distance to the class means, one shared Ledoit-Wolf covariance |
| S6 | nearest-neighbour distance: mean distance to the 5 nearest training embeddings |

**Measures.** Per score (on SVHN): AUROC (outside images positive), AUPR-out and FPR at 95% TPR, with a
stratified bootstrap 95% interval for the AUROC (2,000 resamples, seed 42). Differences between S1 and each
other score use the paired bootstrap over images. **Control:** the S1 AUROC with the labels shuffled 1,000
times (seed 42); it should sit near 0.5.

**Statements** (each is met, not met, or inconclusive; a statement that is not met is reported as it is):

| ID | Statement | Met if |
|---|---|---|
| O1 (RQ3) | the SOM detects outside images | on every outside set, the S1 AUROC interval lies above 0.5 |
| **O2 (RQ4, primary)** | the SOM is competitive with max-softmax | on every outside set, the lower end of the paired interval of S1 minus S3 is above -0.01 |
| O3 (RQ4) | the SOM is competitive with energy, Mahalanobis and nearest neighbours | as O2, for S4, S5 and S6 separately |
| O4 | the U-Matrix detects outside images | on every outside set, the S2 AUROC interval lies above 0.5 |

With one outside set (section 2), "on every outside set" means "on SVHN". There is one network and one map
here, so no claim of "better" is made: the group proposal (RQ4) requires at least 5 seeds for that.

**Further Phase 3 steps, as on MNIST** (`PREREGISTRATION_qe_novelty.md` and the novelty-benchmark and U-Matrix
documents): the QE flag at the 95th percentile of the training QE with its detection and false-alarm rates;
the threshold sweep; the random-image control; the classical baselines (Isolation Forest, One-Class SVM and
the autoencoder, with standardised inputs as in MNIST's final setup); the same-budget baseline comparison; the
check of how much QE depends on the embedding length; and the U-Matrix figures and outside-set overlays.

**Not repeated:** MNIST steps whose only job was to re-create or audit MNIST's own earlier runs (the historical
baseline verification, the historical SVHN inspection, the U-Matrix and random-control reproductions, the
provenance audit, the 2026-11-02 reorganisation and the evidence consolidation). Fashion-MNIST has no earlier
runs to re-check. Diagnostics that MNIST added to explain a problem it found (the autoencoder scaling checks
and the SVHN length diagnostic) are run only if the same problem appears, and that is recorded in
`../DEVIATIONS.md`.

*(Added 2026-11-21, amendment A2, before any Fashion-MNIST Phase 3 run and before any novelty score
existed. It fills in what this section left unsaid; no rule above changes.)*

- **Verdicts.** O1 and O4: *met* if the AUROC interval lies above 0.5; *not met* if it lies at or below
  0.5; *inconclusive* if it contains 0.5. O2 and O3, with d = AUROC(S1) minus AUROC(other) and its paired
  interval: *met* if the lower end is above -0.01; *not met* if the upper end is below -0.01;
  *inconclusive* otherwise. O3 gets one verdict for each of S4, S5 and S6.
- **Where O1 to O4 are judged.** One step, `novelty_statements`, computes all six scores on the same
  10,500 validation and 26,032 SVHN images in one run, with the MNIST Phase 3 functions, and applies the
  rules above: AUROC with its stratified bootstrap interval, AUPR-out with its chance level, FPR at 95%
  TPR, the paired bootstrap of S1 minus each other score (2,000 resamples, seed 42), and the label-shuffle
  control for S1 (1,000 shuffles, seed 42). The paired comparisons in the other steps (QE against
  max-softmax, energy, nearest neighbours and the U-Matrix) are reported as supporting results; the
  statements are taken from `novelty_statements` only. It is the only step with QE against Mahalanobis.
  The same step also reports the group proposal's RQ3 and RQ4 (`proposals/Group_proposal.md`), by the same two
  rules: RQ3 as O1; RQ4 as O2 but against the best of S3 to S6 by AUROC, as the proposal defines it. O1 to
  O4 remain this protocol's statements; RQ3 and RQ4 are reported so both datasets answer the same questions.
- **Which MNIST steps run.** `qe_random_control` (are QE-flagged validation images more often CNN errors
  than random picks?) and `qe_u_matrix` (QE against the U-Matrix on validation) re-created earlier runs on
  MNIST, which is why they are called reproductions above. Fashion-MNIST has no earlier runs, so here they
  are first runs of those analyses, and they run: the first is the random-image control listed above, and
  the second gives the numbers the U-Matrix validation figure is checked against.
  `qe_score_distribution_figure` is not repeated: it re-ran MNIST's `qe_novelty` only because the first run
  did not keep its figure, and the Fashion-MNIST `qe_novelty` run keeps it.
- **Order.** `qe_novelty` (the frozen training-QE threshold) first, then `qe_ood`, whose metrics the
  five-method benchmark must reproduce, then the benchmark and the steps that read its score files.

## 4. Near novelty (held-out classes)

`src/v1_mnist/docs/p3_novelty_detection/held_out_class_novelty/HELD_OUT_CLASS_NOVELTY_PROTOCOL.md`
unchanged, with Fashion-MNIST classes. The five splits are the same five sets of class numbers, drawn by the
same rule (numpy `default_rng(2026)`, 4 unknown classes per split), so they are not chosen with knowledge of
the clothing classes: [0, 1, 5, 6], [2, 3, 5, 8], [1, 5, 6, 7], [2, 5, 6, 8], [0, 4, 5, 8]. One network and
one map per split, recipes as in the Phase 1 and Phase 2 protocols (section 1). Statements H1, H2 (primary:
SOM better than max-softmax) and H3 unchanged. Training and validation images only.

## 5. Leakage control

- The SOM scaler and map, the Mahalanobis means and covariance, the nearest-neighbour reference and the
  thresholds are all fitted or chosen on training or validation data.
- The outside set is used only to compute novelty scores; nothing is fitted or chosen on it.
- The held-out-class networks never see their unknown classes, and their test split is never read.
- Every run records the SHA-256 of its inputs; a finished run is never overwritten.
- **Declared:** this protocol was written after the MNIST results were known. It copies MNIST's design
  instead of adapting it to those results, and no Fashion-MNIST result existed when it was written.

## 6. Limitations stated before the run

- One CNN and one SOM, as for MNIST; seed-to-seed variation is measured only in the layer ablation
  (SOM seeds) and Phase 4 (CNN seeds).
- If more than one outside set is used, they differ from Fashion-MNIST in different ways; results are
  reported per set and not averaged. *(Added 2026-11-21, amendment A1:)* only SVHN is used, so O1 to O4
  speak to far novelty only. A high AUROC on SVHN does not show that the scores detect outside images
  that look like clothing; near novelty is covered only by the held-out classes, which come from the same
  dataset.
- The held-out-class splits share classes, so they are not independent.

## 7. Recorded outcome

Filled in after the runs, without changing sections 1 to 6. Recorded 2026-11-21. One LeNet-5 and one SOM
(15x15, seed 42) unless stated; Fashion-MNIST validation (10,500 images) against SVHN test (26,032 images);
the Fashion-MNIST test split was never read. Every run folder is listed in
`outputs/v1_fashion_mnist/p3_novelty_detection/run_manifest.csv`.

**Far novelty: SVHN** (`novelty_benchmark/fashion_mnist_svhn_novelty_statements_s42_v1`). AUROC with its
stratified bootstrap 95% interval:

| Score | AUROC | Paired S1 minus this score |
|---|---|---|
| S4 energy | 0.982 [0.981, 0.984] | -0.193 [-0.199, -0.186] |
| S3 max-softmax | 0.884 [0.880, 0.888] | -0.094 [-0.101, -0.088] |
| S6 nearest neighbours | 0.864 [0.859, 0.869] | -0.074 [-0.078, -0.070] |
| S2 U-Matrix | 0.812 [0.805, 0.818] | -0.022 [-0.030, -0.014] |
| S1 SOM QE | 0.789 [0.783, 0.796] | |
| S5 Mahalanobis | 0.713 [0.706, 0.720] | +0.076 [+0.071, +0.082] |

| Statement | Verdict |
|---|---|
| O1 (RQ3): the SOM detects outside images | **met** |
| **O2 (RQ4, primary): the SOM is competitive with max-softmax** | **not met** |
| O3: competitive with energy / Mahalanobis / nearest neighbours | not met / met / not met |
| O4: the U-Matrix detects outside images | **met** |
| Group proposal RQ3 / RQ4 (RQ4 against the best of S3 to S6, energy) | met / not met |

Control: with the labels shuffled 1,000 times the S1 AUROC averaged 0.500 (largest 0.510). The QE-against-
max-softmax interval of `qe_msp_paired_comparison` equals the one above to every printed digit.

**The frozen QE flag fails on SVHN.** At the 95th percentile of the training QE (1.4215) the flag marks 5.1%
of the validation images but only 3.0% of SVHN (792 of 26,032); on MNIST the same rule caught 96.2%. The
threshold sweep (`fashion_mnist_svhn_qe_threshold_sweep_s42_v1`) shows the SVHN detection rate below the
validation false-alarm rate at every cut-off from the 95th percentile up; only the 90th percentile flags
slightly more SVHN than validation images (13.1% against 10.1%). The SOM ranks SVHN above most validation
images (AUROC 0.789), but SVHN sits in the middle of the QE range, below the validation images' upper tail.

**Embedding length** (`fashion_mnist_svhn_qe_norm_dependence_s42_v1`, exploratory as on MNIST). SVHN
embeddings are much shorter than Fashion-MNIST's (mean length 8.5 against 20.7; 95% of SVHN below 11.7,
95% of validation above 13.2). Among the 532 SVHN images whose length lies in the validation 5-95% band,
the QE AUROC is 0.958 [0.953, 0.962]. This fits the explanation that SVHN's short embeddings are what hide it
from QE, but the 532 images are a small, selected subset, so it is not shown.

**Classical baselines** (`fashion_mnist_svhn_five_method_comparison_s42_v1`): Isolation Forest 0.266, One-
Class SVM 0.734, autoencoder 0.583; the autoencoder with standardised inputs
(`fashion_mnist_svhn_ae_train_standardized_s42_v1`) 0.676. Isolation Forest below 0.5 ranks SVHN as more
normal than Fashion-MNIST, as on MNIST (0.238), and the autoencoder again changes strongly with its input
scaling and training size. Both are the problems the two MNIST diagnostics in section 3 were written for, so
by section 3 they ran (`../DEVIATIONS.md`, 2026-11-21; exploratory, as on MNIST):

- *Training size and scaling* (`fashion_mnist_svhn_autoencoder_scaling_diagnostic_s42_46_v1`, seeds 42-46,
  mean ± standard deviation). Autoencoder: raw inputs 0.395 ± 0.030 (10,000 training rows) and 0.553 ± 0.028
  (49,000); standardised inputs 0.742 ± 0.089 and 0.716 ± 0.069. Isolation Forest (standardised): 0.272 ±
  0.036 and 0.244 ± 0.054. The autoencoder depends mainly on input scaling, not on training size (as on
  MNIST, where the effect was far larger), and Isolation Forest stays below chance at both sizes, so its
  reversal is not caused by too little training data. Both match the unified run's seeds (largest AUROC
  difference 0.0009).
- *Embedding length.* Scoring by length alone ("shorter is more novel") separates SVHN from Fashion-MNIST
  almost perfectly: AUROC 0.995 (the run reports 0.005 for "longer is more novel",
  [0.0046, 0.0056]). Inverting SVHN's grey levels, so that its images are light on dark like
  Fashion-MNIST, lengthens the embeddings only slightly (mean 8.50 to 9.18; longer for 61% of the images;
  `fashion_mnist_svhn_inverted_embedding_norm_s42_v1`), so the brightness direction explains little of the gap
  to Fashion-MNIST's 20.7.

So SVHN differs from Fashion-MNIST in one simple feature, the embedding length, which a trivial rule already
uses. A far-novelty AUROC on SVHN therefore mostly measures how strongly a score follows that length.
Isolation Forest, which treats the short SVHN embeddings as typical, is reversed; QE follows it only weakly
(AUROC 0.789), which fits the norm-matched result above. Why the network gives SVHN short embeddings is not
shown by these runs.

**Near novelty: held-out classes** (`held_out_class_novelty/held_out_class_novelty_lenet5_som15x15_nb11_ep250_s42`).
Per split, a new LeNet-5 and SOM never see 4 classes; QE minus each score, mean and 95% t-interval over the
5 splits:

| Statement | Result | Verdict |
|---|---|---|
| H1: the SOM detects the held-out classes | QE AUROC 0.739 to 0.831; every interval above 0.5 | **shown** |
| **H2 (primary): the SOM is better than max-softmax** | **+0.046 [+0.034, +0.057]**, larger in all 5 splits | **shown** |
| H3: not worse by more than 0.01 than max-softmax / energy / Mahalanobis | +0.046 / +0.011 [-0.041, +0.064] / -0.028 [-0.071, +0.014] | shown / inconclusive / inconclusive |
| H4: the map adds to nearest neighbours | -0.048 [-0.071, -0.026] | **opposite shown** (nearest neighbours better) |

The embedding length alone scores below QE in every split (mean AUROC 0.687 against 0.794), so QE's
advantage over max-softmax is not a length effect. On MNIST, H2 was inconclusive.

**Seed-to-seed variation** (from the Phase 2 feature-layer ablation, which trains 5 SOMs on the same `fc2`
embeddings with seeds 1 to 5): the QE AUROC on SVHN ranges from 0.766 to 0.884 with the SOM seed alone. The
intervals above come from one map and reflect only which images were sampled; they do not include this
variation, which is several times larger. The gap to energy (0.982) remains at every seed.

**What this means.** On Fashion-MNIST the SOM detects both kinds of novelty better than chance. Against
*far* novelty (SVHN) it is clearly behind the network's own scores (O2 not met) and its frozen threshold is
not usable; against *near* novelty (unseen clothing classes) it beats max-softmax in every split (H2 shown),
which it did not show on MNIST. Nearest-neighbour distance on the same embeddings is better than the SOM in
both settings.

## 8. References

Keys in `reports/Latex_report/v1_mnist/Final_report/references.bib` and the Phase 3 `references.bib`.

- Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B. and Ng, A. Y. (2011). Reading digits in natural
  images with unsupervised feature learning. NIPS Workshop on Deep Learning and Unsupervised Feature
  Learning. (`netzer2011reading`)
- Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D. and Lakshminarayanan, B. (2019). Do deep generative
  models know what they don't know? ICLR. (`nalisnick2019deep`)
- The score and baseline references as in the MNIST Phase 3 protocols.
