# Protocol: novelty detection on held-out classes (Phase 3, near novelty)

Status: **written 2026-11-17, before any network or map of this step is trained.** Commit this file and
`configs/held_out_class_novelty.yaml` before running. The run records the SHA-256 of both, and every
later call stops if either has changed.

## 1. Why this exists

Phase 3 tested novelty detection with SVHN as the novel set. SVHN is "far" novelty: a different source,
colour and style. Phase 3 also found that SVHN's `fc2` embeddings are much shorter than MNIST's, and that
embedding length alone separates SVHN almost perfectly (AUROC 0.9999). So the SVHN result cannot tell
whether the SOM detects new *kinds of images* or only short embeddings.

Instructor Review #5 (2026-10-19) asked for "one near-OOD pair (e.g. MNIST vs held-out MNIST classes)",
because that is "the setting in which a topological method could actually win": the novel images come
from the same source, scanner and writing style as the known ones, and only their class is new.
This is the open-set recognition setting (Scheirer et al.; Neal et al.).

Questions:

1. **Does the SOM's quantization error (QE) flag digits of classes the network never saw?**
2. **Does it do better than the network's own confidence (max-softmax)**, which is the strong baseline in
   open-set recognition (Vaze et al.)? And how does it compare with energy, Mahalanobis and plain
   nearest-neighbour distance?
3. **Control:** is the separation explained by embedding length, as for SVHN?

## 2. Class splits (fixed now)

The standard open-set protocol for MNIST uses 6 known and 4 unknown classes, repeated over 5 random
splits (Neal et al.). We use the same sizes. The 5 splits were drawn once with NumPy
`default_rng(2026)`: draw 4 of the 10 digits without replacement, sort them, and keep the draw if it
differs from the earlier ones, until there are 5. The code re-draws them and stops if the config
differs.

| Split | Unknown (novel) digits | Known digits |
|---|---|---|
| 1 | 0, 1, 5, 6 | 2, 3, 4, 7, 8, 9 |
| 2 | 2, 3, 5, 8 | 0, 1, 4, 6, 7, 9 |
| 3 | 1, 5, 6, 7 | 0, 2, 3, 4, 8, 9 |
| 4 | 2, 5, 6, 8 | 0, 1, 3, 4, 7, 9 |
| 5 | 0, 4, 5, 8 | 1, 2, 3, 6, 7, 9 |

The draw is random, so the digits are not balanced: 5 is unknown in four splits and 9 in none. This is
accepted, not corrected, because choosing splits by hand would be a choice made by us.

## 3. Data (the project's frozen split; the MNIST test split is never read)

For each split:

- **Training:** the training-split images (49,000 in total) of the 6 known digits, labels renumbered
  0 to 5 in increasing digit order.
- **Validation, known digits:** the validation-split images of the 6 known digits. Used (a) to choose the
  network's best epoch by accuracy, exactly as in Phase 1, and (b) as the **normal** images.
- **Validation, unknown digits:** the validation-split images of the 4 unknown digits: the **novel**
  images. Never used for training, epoch choice, fitting or thresholds.
- Images are looked up by the sample IDs Phase 1 saved (`utils/frozen_split.py`), so the split is the
  same one every other phase uses.

## 4. Networks and maps (no tuning)

- **Network:** LeNet-5 with 6 outputs, trained from scratch with the Phase 1 recipe
  (`configs/cnn_baseline.yaml`: Adam, learning rate 0.001, weight decay 0.0001, StepLR 10/0.1,
  20 epochs, batch 64), training seed 42, best epoch by validation accuracy on known digits
  (`utils/cnn_retraining.py`, unchanged recipe).
- **Embeddings:** the `fc2` layer (84 numbers), as in Phase 1.
- **Map:** a 15x15 SOM with the Phase 2 recipe (`configs/som.yaml`: neighbourhood 11, 250 epochs,
  250 steps, seed 42), trained on the training embeddings, with the MinMax scaler to [-1, 1] fitted on
  the training embeddings (as in the SOM re-analysis).
- Nothing is tuned in this step. One network and one map per split.

## 5. Scores (a higher score means "more novel")

Anything fitted is fitted on the split's **training** embeddings only.

| ID | Score | Definition |
|---|---|---|
| S1 | SOM quantization error (QE) | distance from the scaled embedding to its best-matching neuron |
| S2 | max-softmax | 1 minus the largest softmax probability (Hendrycks and Gimpel) |
| S3 | energy | minus log-sum-exp of the logits, temperature 1 (Liu et al.) |
| S4 | Mahalanobis | smallest squared Mahalanobis distance to the known-class means, one shared Ledoit-Wolf covariance (Lee et al.; Ledoit and Wolf), as in Phase 3 |
| S5 | nearest-neighbour distance | mean Euclidean distance to the 5 nearest training embeddings (Sun et al.), as in Phase 3 |
| D1 | embedding length (diagnostic only) | minus the L2 norm of the `fc2` embedding; reported to check the length effect, not compared as a detector |

S1 to S4 and D1 use the functions of the novelty-after-retraining step (`novelty_after_retraining/scoring.py`),
which call Phase 3's functions unchanged; S5 uses Phase 3's `knn_baseline.py`.

## 6. Measures

Per split and score: AUROC (unknown digits as the positive class), AUPR-out and FPR at 95% TPR (Phase 3's
`ood_metrics.py`), with a stratified bootstrap 95% interval for the AUROC (2,000 resamples, seed 42).
Within a split, the difference between S1 and each other score uses Phase 3's paired bootstrap over images.
For S1 to S4 also the detection and false-alarm rates at the 95th percentile of the training scores
(reported, not judged).

**Control (null).** Per split, the S1 AUROC with the normal/novel labels shuffled 1,000 times (seed 42),
Phase 3's `permutation_auroc_null`. It should sit near 0.5.

**Where the novel digits land (descriptive).** Per split, the number of map cells that together hold 90%
of the novel images, and the same for the normal images.

**Length check (descriptive).** D1's AUROC and the Spearman correlation between QE and embedding length.

## 7. Statements (fixed now)

Across splits, a difference is taken **per split** (5 values); we report their mean and a 95% t-interval
(4 degrees of freedom), as in the Phase 4 comparisons. Each statement has three possible outcomes:
shown, opposite shown, or inconclusive (Walker and Nowacki). The margin of 0.01 AUROC is the project's
"competitive" margin (`reports/Proposal.md`, RQ4), fixed on 2026-11-10, before this step existed.

| Statement | Shown if | Opposite shown if |
|---|---|---|
| H1: the SOM detects held-out digits | in every split, the S1 AUROC interval lies above 0.5 | in any split, the interval includes or lies below 0.5 (the splits are listed) |
| **H2 (primary): the SOM is better than max-softmax** | S1 minus S2: lower end of the interval above 0 | upper end below 0 (max-softmax better) |
| H3: the SOM is not worse than max-softmax, energy or Mahalanobis by more than 0.01 | for each of S2, S3, S4 separately: S1 minus that score, lower end above -0.01 | upper end below -0.01 |
| H4: the map adds to plain nearest-neighbour distance | S1 minus S5: lower end above 0 | upper end below 0 |

H2 is the primary statement; H1, H3 and H4 are secondary and are reported without correction for
multiple comparisons. A statement that is not met is a result and is reported as it is.

Also reported, not judged: per split, whether S1 is "competitive" by the RQ4 rule (lower end of the
paired bootstrap interval of S1 minus the best of S2 to S4 above -0.01); the network's closed-set
accuracy on known digits (Vaze et al. link open-set performance to closed-set accuracy).

## 8. Limitations (stated before the run)

- **One dataset, one network size.** MNIST only; LeNet-5 only. Not claimed to transfer.
- **Few splits, one seed each.** 5 splits, each with one training seed and one SOM seed. The t-interval
  treats the splits as exchangeable; they share images and some unknown digits, so they are not fully
  independent, and the interval may be too narrow.
- **Unbalanced splits** (section 2).
- **Validation images used twice.** Known-digit validation images choose the best epoch (by accuracy only)
  and are the normal set, as in Phase 3, where the Phase 1 network's epoch was also chosen on validation.
- **Not the published split files.** The 5 splits are drawn by us with the published sizes; numbers are
  not directly comparable with tables in the open-set literature.

## 9. Leakage control

- Novel (unknown-digit) images are never used for training, epoch choice, fitting, thresholds or any
  setting. All settings come from Phase 1 and Phase 2 configs and are fixed before the run.
- Every fitted quantity (scaler, SOM, Mahalanobis, nearest neighbours, thresholds) uses training
  embeddings only.
- The MNIST test split is not read (only `load_train_val_images` is called).

## 10. Runs and records

- Run ID: `held_out_class_novelty_lenet5_som15x15_nb11_ep250_s42[_<revision>]`. An existing run is never
  overwritten; a change needs a new revision in the config.
- Per split (folder `splits/split<k>/`): the checkpoint, embeddings, SOM and score arrays (DVC, not git),
  and a small `metrics.json` (git), written last, so a split counts as done only when it exists.
- `results.json` and the tables (git) once all 5 splits are done.
- Expected time on EC2: about 10 to 20 minutes per split (training one LeNet-5 and one SOM), so about
  1 to 2 hours in total.

## 11. Recorded outcome

[TBD: filled in after the run, from `results.json`; sections 1 to 10 stay unchanged.]

## 12. References

- Hendrycks, D., and Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR*.
- Ledoit, O., and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis* 88(2), 365-411.
- Lee, K., Lee, K., Lee, H., and Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *NeurIPS*.
- Liu, W., Wang, X., Owens, J., and Li, Y. (2020). Energy-based out-of-distribution detection. *NeurIPS*.
- Neal, L., Olson, M., Fern, X., Wong, W.-K., and Li, F. (2018). Open set learning with counterfactual images. *ECCV*, 613-628.
- Scheirer, W. J., de Rezende Rocha, A., Sapkota, A., and Boult, T. E. (2013). Toward open set recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 35(7), 1757-1772.
- Sun, Y., Ming, Y., Zhu, X., and Li, Y. (2022). Out-of-distribution detection with deep nearest neighbors. *ICML*.
- Vaze, S., Han, K., Vedaldi, A., and Zisserman, A. (2022). Open-set recognition: A good closed-set classifier is all you need? *ICLR*.
- Walker, E., and Nowacki, A. S. (2011). Understanding equivalence and noninferiority testing. *Journal of General Internal Medicine* 26(2), 192-196.
