# Protocol: novelty detection after retraining (Phase 4)

Status: **written 2026-11-17, before any score is computed.** Commit this file and
`configs/novelty_after_retraining.yaml` before running. The run records the SHA-256 of both, and the
later stages stop if either has changed.

## 1. Why this exists

The proposal's results table (Phase 5, week 12) lists "baseline accuracy, post-retraining accuracy,
novelty detection AUROC". The Phase 4 report has the first two; the novelty AUROC after retraining is
marked [TBD]. This step fills it in.

Questions:

1. **Does SOM-guided retraining change how well the SOM detects novel inputs?** Novelty is measured as
   in Phase 3: MNIST validation images are normal, SVHN test images are novel.
2. **Does it change the SOM's gap to the standard detectors** (max-softmax, energy, Mahalanobis) on the
   same networks?
3. **Control:** Phase 3 found that much of the SOM's separation of SVHN comes from SVHN's much shorter
   embeddings. Is that still true after retraining?

## 2. Inputs (read-only)

- **Cells:** the same 15 cells as the SOM re-analysis: arms `standard`, `augmented_som` and
  `augmented_random` of the targeted augmentation run (`configs/retraining.yaml`, section
  `augmentation`), seeds 1 to 5.
- For every cell, from the SOM re-analysis run (`configs/som_reanalysis.yaml`):
  - the retrained LeNet-5 checkpoint (SHA-256 checked against the value recorded after training);
  - the cell's new 15x15 SOM (loaded, not retrained; SHA-256 checked against the re-analysis records);
  - the cell's training and validation `fc2` embeddings (reused; re-extraction must reproduce them).
- **Reference:** the Phase 1 network and the Phase 2 SOM (`som_15x15_nb11_ep250_s42_v2`), scored by
  the same code, so every number is computed the same way.
- **Normal images:** the 10,500 MNIST validation images of the frozen split.
- **Novel images:** the 26,032 SVHN test images from the Phase 3 shared provenance file
  (`data/ood_svhn/test_32x32.mat`, SHA-256 checked), with Phase 3's fixed transform: grayscale,
  bilinear resize to 28x28, scaled to [0, 1]. Never downloaded, never used for fitting.
- **Not read:** the MNIST test split.

## 3. Scores (a higher score means "more novel")

Each cell is scored with its own network and its own SOM. Anything fitted is fitted on that cell's
**training** embeddings only.

| ID | Score | Definition |
|---|---|---|
| S1 | SOM quantization error (QE) | distance from the embedding (MinMax-scaled to [-1, 1] with the scaler fitted on the cell's training embeddings, as in the re-analysis) to its best-matching neuron |
| S2 | max-softmax | 1 minus the largest softmax probability (Hendrycks and Gimpel) |
| S3 | energy | minus log-sum-exp of the logits, temperature 1 (Liu et al.) |
| S4 | Mahalanobis | smallest squared Mahalanobis distance to the class means, with one shared covariance fitted on the training embeddings (Lee et al.; one layer, without input perturbation). As in Phase 3, the covariance is the Ledoit-Wolf shrinkage estimate (Ledoit and Wolf), not the plain sample covariance Lee et al. used; shrinkage keeps the 84x84 estimate well conditioned |
| D1 | embedding length (diagnostic only) | minus the L2 norm of the `fc2` embedding (shorter means more novel); reported to check the length effect, not compared as a detector |

S2 to S4 use Phase 3's functions (`novelty_benchmark/required_baselines.py`) unchanged.

## 4. Measures

For every cell and score: AUROC (SVHN as the positive class), AUPR-out and FPR at 95% TPR, with Phase 3's
functions (`qe_novelty_detection/ood_metrics.py`). For S1 also the detection rate and the false-alarm rate at
the 95th percentile of the cell's training QE (Phase 3's fixed threshold rule); reported, not judged.
For D1: Spearman correlation between QE and embedding length over all scored images.

## 5. Comparisons and statements (fixed now)

Differences between arms are taken **per seed** (the same CNN seed in both arms): 5 paired differences,
their mean, and a 95% t-interval (4 degrees of freedom), as in the SOM re-analysis. Within one cell, the
difference between two scores uses Phase 3's paired bootstrap over images (2,000 resamples, seed 42,
stratified by normal/novel).

Each statement has **three possible outcomes**, as in non-inferiority testing (Walker and Nowacki):
the interval can show the property, show the opposite, or be too wide to tell. With 5 seeds, "too wide
to tell" is a real possibility and is reported as **inconclusive**, never as a failure or a success.

| Statement | Shown if | Opposite shown if | Answers |
|---|---|---|---|
| N1: the SOM still detects novelty after retraining | in every one of the 10 retrained `augmented_som` and `augmented_random` cells, the S1 AUROC 95% bootstrap interval lies above 0.5 (Phase 3's RQ3 rule) | in any of those cells the interval lies below or includes 0.5 (reported per cell) | 1 |
| N2: SOM-guided retraining does not harm the SOM's novelty detection | S1 AUROC, `augmented_som` minus `standard`: lower end of the interval above -0.01 | upper end below -0.01 (harmed by more than the margin) | 1 |
| N3: SOM guidance itself does not harm it | as N2, `augmented_som` minus `augmented_random` | as N2 | 1 |
| N4: the gap to max-softmax does not widen | (S1 minus S2 AUROC), `augmented_som` minus `standard`: lower end of the interval above -0.01 | upper end below -0.01 | 2 |

**The margin.** -0.01 AUROC is not taken from the literature. It is the proposal's own "competitive"
margin for RQ4 (`reports/Proposal.md` section 6), reused so that the project has one margin. That margin
was fixed on 2026-11-10, after the Phase 3 development results were seen (`DEVIATIONS.md`); it was not
fixed after any result of this step.

Also reported, not judged: S1 minus S3 and S1 minus S4 per cell with their bootstrap intervals; the
count of cells where S1 is "competitive" by the proposal's RQ4 rule (lower end of the paired bootstrap
interval for S1 minus the best of S2 to S4 above -0.01); the reference model's values; D1 per arm.

A statement that is not met is a result and is reported as it is. Nothing here changes a Phase 3 or
Phase 4 result or criterion.

## 6. Limitations (stated before the run)

- **Far novelty only.** SVHN is a "far" novel set for MNIST (different source, colour and style); the
  surveyed literature distinguishes far from near novelty, where detectors behave differently (Yang et al.).
  A near-novelty set (for example Fashion-MNIST images against an MNIST network) is not part of this step.
- **The length effect.** Phase 3 found that SVHN's `fc2` embeddings are much shorter than MNIST's and that
  much of QE's separation follows from that. A related effect of SVHN being rated as "more normal" by
  likelihood models is reported by Nalisnick et al. D1 is reported to see whether retraining changes this;
  it does not remove the effect.
- **Variation.** One SOM seed per cell (the re-analysis maps) and 5 CNN seeds per arm, so only large
  differences can be shown; the paired design follows Bouthillier et al.
- **One dataset.** All numbers are MNIST; they are not claimed to transfer to other datasets.

## 7. Leakage control

- Fitting (SOM scaler, Mahalanobis means and covariance) uses each cell's training embeddings only. The
  SOMs and networks are reused, not retrained or tuned.
- MNIST validation images are used as normal images only; their labels are not used by any score.
- SVHN is used for scoring only. No threshold, setting or score direction is chosen from SVHN results
  (Phase 3 already saw SVHN; this step adds no choice that depends on it).
- The MNIST test split is not read.

## 8. Runs and records

- Run ID: `novelty_after_retraining_<selected SOM run ID>[_<revision>]`. An existing run is never
  overwritten; a change needs a new revision in the config.
- Per cell: the five scores for every scored image (DVC, not git) and a small metrics JSON (git).
  Tables: AUROC/AUPR/FPR per cell and score, the paired comparisons, the statements N1 to N4.
- Expected time on EC2: a few minutes per cell (one pass of 36,532 images through LeNet-5 and one SOM
  assignment); under one hour in total.

## 9. Recorded outcome

Filled in after the run, without changing sections 1 to 8. Each statement is recorded as shown, opposite shown or inconclusive.

| Statement | Result |
|---|---|
| N1 | [TBD] |
| N2 | [TBD] |
| N3 | [TBD] |
| N4 | [TBD] |

## 10. References

- Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems (MLSys)* 3.
- Hendrycks, D., and Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR*.
- Ledoit, O., and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis* 88(2), 365-411.
- Lee, K., Lee, K., Lee, H., and Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *NeurIPS*.
- Liu, W., Wang, X., Owens, J., and Li, Y. (2020). Energy-based out-of-distribution detection. *NeurIPS*.
- Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D., and Lakshminarayanan, B. (2019). Do deep generative models know what they don't know? *ICLR*.
- Walker, E., and Nowacki, A. S. (2011). Understanding equivalence and noninferiority testing. *Journal of General Internal Medicine* 26(2), 192-196.
- Yang, J., Zhou, K., Li, Y., and Liu, Z. (2024). Generalized out-of-distribution detection: A survey. *International Journal of Computer Vision* 132, 5635-5662.

