# Capstone Proposal: Where Do Neural Networks Fail?

## Post-Training Analysis and Novelty Detection Using Neural-Network Self-Organizing Maps (NNSOM)

|                |                                                                 |
| -------------- | --------------------------------------------------------------- |
| **Course**     | DATS 6501 Capstone, Fall 2026, The George Washington University |
| **Group**      | Group 5                                                         |
| **Team**       | Fyrooz Anika Khan, Nazish Atta                                  |
| **Advisor**    | Dr. Amir Jafari                                                 |
| **Repository** | https://github.com/nazishatta/fall-2026-group5                  |
| **Version**    | Group proposal, revised 2026-11-22                              |

The advisor's original topic description is kept in [`Project_proposal.md`](Project_proposal.md).
This document is the group's own proposal: what we will do, with which data, how we will judge
success, who does what, and when.

---

## 1. Summary

A neural network can score 99% accuracy and still fail in a systematic way. The mistakes are
not random. They cluster around certain kinds of inputs. Common explanation tools (saliency
maps, LIME, SHAP) explain **one prediction at a time**. They do not show **where in the model's
internal representation the failures gather**.

We use a **Self-Organizing Map (SOM)** to get that population-level view. A SOM is an
unsupervised method that squeezes high-dimensional data onto a flat grid so that similar
items land near each other. We train a CNN, take its internal features (embeddings), train a
SOM on them with the NNSOM library, and then ask three things:

1. **Where** on the map do the model's mistakes concentrate?
2. Can the SOM **flag inputs the model has never seen anything like** (novelty detection)?
3. Can the SOM **tell us which samples to retrain on**, and does that beat picking samples at random?

**Central hypothesis.** SOM-based topology and quantization signals expose structured regions
of model failure; they detect distributional novelty; and SOM-derived diagnostics guide
retraining that measurably improves the model or its representation. We treat this as a
hypothesis to test, not a result to defend. Parts that the evidence does not support will be
reported as unsupported.

### Where we are today (2026-11-22): results on MNIST and Fashion-MNIST

Both datasets have been through all four phases with the same code, SOM settings and
statistics, and a protocol written before each run. Details are in the phase reports in
`reports/Latex_report/v1_mnist/` and `reports/Latex_report/v1_fashion_mnist/`. The first part is
MNIST, the second part is Fashion-MNIST.

**MNIST**

- **Phase 1, CNN baseline.** LeNet-5 gets **99.18%** of the test images right (10,414 of 10,500;
  95% Wilson interval 98.99–99.34%), macro-F1 0.9918. It makes 92 mistakes on the 10,500
  validation images.
- **Phase 2, the map (RQ1).** A 15x15 SOM (`som_15x15_nb11_ep250_s42_v2`) on the 84-dimensional
  fc2 embeddings. Validation quantization error 1.041, topographic error 6.18%. The map's cells
  follow the digit classes: purity 98.19%, against 17.4% with shuffled labels (p = 0.001).
- **Phase 2, where mistakes gather (RQ2): supported.** The 11 hotspot cells hold **11.1 times**
  their fair share of the validation mistakes (95% range 6.1–17.6; p = 0.005), and the result
  holds on the 10x10 (10.5) and 20x20 (13.4) maps too. Cells where digits mix have more mistakes
  (Spearman +0.82). But as a mistake detector a SOM cell is no better than a k-means cell
  (AUROC 0.685 vs 0.693), and the CNN's own confidence (max-softmax) is far better (0.981). The
  SOM's hotspots are tighter: 1.6% of the images hold 18% of the mistakes (k-means: 3.8% and 20%).
- **Phase 3, novelty detection (RQ3, RQ4).** With MNIST validation as normal data and SVHN as
  novel data, the SOM's quantization error separates them well (AUROC **0.981**, 95% interval
  0.979–0.983), so **RQ3 is supported** for this pair. But energy (0.9998) and max-softmax
  (0.998) rank better; the paired QE minus max-softmax difference is −0.017 (−0.019 to −0.015),
  outside the −0.01 margin, so **RQ4 ("competitive") is not supported**. Much of QE's separation
  of SVHN is tied to SVHN's much shorter embeddings. The MNIST test split was not used in Phase 3.
- **Phase 4, guided retraining (RQ5): supported for augmentation.** Over 5 seeds, adding mildly
  augmented copies of SOM-chosen training images lowers the test error on hotspot images from
  11.4% to **7.8%**. That beats random augmentation by 2.5 points (95% interval 0.7–4.5), with
  the macro-F1 limit held. It ties with max-softmax-guided augmentation (difference −0.7 points,
  −2.9 to +1.2). Ordering training by SOM difficulty (curriculum) makes things worse (hotspot
  error 21.4%).
- **Phase 4, the map after retraining (RQ6): not supported.** Purity and topographic error do not
  change beyond seed-to-seed noise. The hotspot error rate and the overall validation error do
  drop after SOM-guided augmentation.
- **In short (MNIST):** the SOM does not beat the classifier's own confidence at detecting
  mistakes or novel inputs. What it adds is a map of **where** mistakes and novel inputs land, and
  choosing retraining images from that map works as well as choosing them by confidence.

**Fashion-MNIST**

- **Phase 1, CNN baseline.** The same LeNet-5 recipe, trained once and not tuned, gets **90.90%**
  of the test images right (9,544 of 10,500; 95% Wilson interval 90.33–91.43%), macro-F1 0.909.
  It makes 876 mistakes on the validation images and 956 on the test images (MNIST: 92 and 86),
  mostly between upper-body garments; Shirt is the weakest class (F1 0.746).
- **Phase 2, the map (RQ1) and where mistakes gather (RQ2): supported.** The 15x15 map's weighted
  purity is 0.893, against 0.173 with shuffled labels (p = 0.001). The 11 hotspot cells hold
  **3.9 times** their fair share of the validation mistakes (95% range 3.2–4.4; p = 0.005), and
  the result holds on the 10x10 and 20x20 maps. The ratio is smaller than on MNIST because a
  ratio cannot exceed one divided by the overall error rate (about 12 here, about 114 on MNIST).
  As on MNIST, a SOM cell is no better than a k-means cell at finding mistakes (AUROC 0.811 vs
  0.812) and the CNN's own confidence is better (0.906). The pre-registered grid comparison finds
  15x15 better than 20x20 at locating mistakes (−0.042, 95% range −0.077 to −0.008).
- **Phase 3, novelty detection (RQ3, RQ4).** With SVHN as novel data, the SOM's quantization
  error gives an AUROC of **0.789** (0.783–0.796), so **RQ3 is supported**. Energy (0.982) and
  max-softmax (0.884) rank better, and the paired difference to max-softmax is −0.094
  (−0.101 to −0.088), so **RQ4 ("competitive") is not supported**. The 95th-percentile threshold
  fitted on training data flags only 3.0% of SVHN. SVHN's embeddings are much shorter than
  Fashion-MNIST's, which a rule on embedding length alone already uses. For near novelty (4 of
  the 10 classes held out, 5 splits) the SOM does better than max-softmax in every split
  (+0.046, 0.034 to 0.057), which it did not show on MNIST; nearest-neighbour distance is still
  better than the SOM.
- **Phase 4, guided retraining (RQ5): not supported for the primary statement.** MNIST's
  augmentation ranges failed the transform check on Fashion-MNIST (accuracy on altered images
  dropped by 0.081; limit 0.02). The rule to halve the ranges was written before re-checking; the
  second attempt also failed (0.029) and the third passed (0.012), so the ranges are four times
  weaker than on MNIST. The SOM pool holds 19,997 images (41% of the training set; MNIST 8.6%).
  Over 5 seeds, SOM-chosen augmentation lowers the hotspot test error from 36.0% to 34.0%, but
  its advantage over random augmentation (−1.0 point, 95% interval −3.2 to +1.1) is not
  distinguishable from zero (MNIST: −2.5 points, met). Ordering training by SOM difficulty
  (curriculum) makes things worse on both datasets, more so here (+3.3 points overall).
- **Phase 4, the map after retraining (RQ6): not supported.** Purity does not improve and
  topographic error is not shown to stay equal. SOM guidance does shrink the hotspot error rate
  more than random guidance does (−3.2 points, −5.2 to −1.2). Novelty detection gets worse after
  retraining (the SOM's AUROC on SVHN falls from 0.838 to 0.715), which it did not on MNIST.
- **In short (both datasets):** the SOM does not beat the classifier's own confidence or energy at
  detecting mistakes or far novel inputs. What it adds is a map of **where** mistakes and novel
  inputs land. Retraining on SOM-chosen images helped on MNIST but not clearly on Fashion-MNIST,
  where the augmentation had to be much weaker and the SOM pool was large.
- **Still to do:** documents only (paper, poster, deck, videos) and repository clean-up; the
  technical freeze was on 2026-11-22, so no new experiments.

---

## 2. Research questions

|         | Question                                                                  | Main output                                             |
| ------- | ------------------------------------------------------------------------- | ------------------------------------------------------- |
| **RQ1** | What does the CNN's feature space look like on a topology-preserving map? | Cluster structure, class overlap, density maps          |
| **RQ2** | Do misclassifications sit in identifiable regions of that map?            | Error-hotspot statistics and example images             |
| **RQ3** | Can SOM-derived scores identify out-of-distribution (OOD) samples?        | AUROC, AUPR, FPR at 95% TPR                             |
| **RQ4** | How does SOM novelty detection compare with established methods?          | Side-by-side table against baselines (Section 5)        |
| **RQ5** | Can SOM error regions pick samples worth retraining on?                   | SOM-guided selection vs. a **random-selection control** |
| **RQ6** | After retraining, is the representation measurably better organized?      | Before/after map metrics                                |

---

## 3. Data

### 3.1 Datasets and access terms

All datasets are public. None contains personal data. All download automatically through
`torchvision`, so no accounts or approvals are needed. Raw data is never committed to git.

| Dataset                    | Role                                                           | Size                                                               | Access and terms                                                                                                                                                                                           |
| -------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MNIST**                  | Main development dataset                                       | 70,000 grayscale 28x28 digit images, 10 classes                    | `torchvision.datasets.MNIST`. Originally distributed from Yann LeCun's site and widely used for research. **We have not re-checked its exact terms; confirm them on that page.** Cite LeCun et al. (1998). |
| **Fashion-MNIST**          | Second dataset: the same pipeline is repeated on it              | 70,000 grayscale 28x28 images, 10 clothing classes                 | `torchvision.datasets.FashionMNIST`. **MIT licence.** Cite Xiao et al. (2017). |
| **SVHN**                   | Novelty (OOD) test set for the MNIST and Fashion-MNIST models; never trained on | 26,032 test digit crops (street-view house numbers)                | `torchvision.datasets.SVHN`. **For non-commercial use only**, which fits this academic project. Cite Netzer et al. (2011). |

We train and analyse on two datasets only: **MNIST and Fashion-MNIST**. SVHN is used only to
test novelty detection, for the models of both datasets. Licence and terms for Fashion-MNIST and SVHN were
checked against each dataset's own page on 2026-09-30. MNIST's terms still need checking (see the table). Please
re-check all of them before any public release of code or results.

### 3.2 Splits and preprocessing

- **Split:** 70% train, 15% validation, 15% test, stratified by class (49,000 / 10,500 / 10,500
  for MNIST). Fixed random seed (42), saved to disk so every experiment uses identical splits.
- **Preprocessing:** pixel values scaled to [0, 1]. Class counts recorded for every split.
- **OOD evaluation:** the model is trained only on the "in-distribution" dataset. OOD sets are
  used **only for evaluation**, never for training, threshold fitting or model selection.
- **Test set discipline:** the test set is not used to choose hyperparameters, SOM grid size or
  thresholds. Those use the validation set.

### 3.3 Data we generate

CNN checkpoints, embeddings (`.npy`), trained SOMs, and figures. Large arrays are **not
stored in git**. They are either regenerated from code with a fixed seed or tracked with DVC
(DVC with the group's S3 remote). Small metrics tables (`.csv`, `.json`) **are**
committed, because those are what a reader needs to check our numbers.

### 3.4 Second dataset

MNIST is good for building the method, but it is near the accuracy ceiling and gives few
mistakes (92 on validation). We therefore repeated the whole pipeline on **Fashion-MNIST**, which
has the same image size and about ten times more mistakes (876 on validation), and report both
datasets in the final results.

---

## 4. Method (the pipeline)

```
data -> CNN baseline -> embeddings (fc2) -> SOM (NNSOM) -> analysis -> novelty detection -> guided retraining
```

The diagram is in the README (`demo/project_pipeline/nnsom_project_pipeline.drawio.svg`).

1. **Baseline.** Train LeNet-5 on MNIST and on Fashion-MNIST with fixed seeds.
   Report accuracy, per-class F1, and the confusion matrix.
2. **Embeddings.** Freeze the trained model. Save the penultimate-layer vector for every sample
   in every split, with sample IDs, labels, predictions and confidence, so results can be traced.
3. **SOM.** Train a 15x15 SOM on training-set embeddings with NNSOM. Choose the grid size using
   the written protocol (`src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/SOM_GRID_SELECTION_PROTOCOL.md`) and
   quantization and topographic error on validation data.
4. **Map analysis (RQ1, RQ2).** Place every sample on its best-matching cell. Measure class
   purity, overlap, and per-cell error rates. Use Wilson confidence bounds on error rates, so
   cells with few samples are not over-interpreted.
5. **Novelty detection (RQ3, RQ4).** Score each sample by its distance to the closest map cell
   (quantization error) and by U-Matrix neighbourhood distance. Compare to baselines.
6. **Guided retraining (RQ5, RQ6).** Retrain using samples chosen from error-hotspot and
   boundary regions. Compare with a **random-selection control of the same size**. Re-extract
   embeddings, retrain the SOM, and compare before and after.

**Reproducibility rules.** Every run has a name that encodes its setup (for example
`fashion_mnist_lenet5_penultimate_som15x15_seed42`). Every reported number can be traced from
config to raw result to table or figure. Unmeasured values are marked as still to be determined, never guessed.

---

## 5. Literature review and state of the art

We group related work into four themes. For each we say what it does, in plain words, and how
this project differs. Full references are in Section 10.

### 5.1 Explaining a model one prediction at a time

- **Saliency maps** (Simonyan et al., 2014) colour the pixels that most change a prediction.
- **LIME** (Ribeiro et al., 2016) fits a small, simple model around one prediction to explain it.
- **SHAP** (Lundberg and Lee, 2017) assigns each input feature a fair share of credit for one
  prediction, using ideas from game theory.

_Difference:_ all three explain a single sample. None shows where failures **concentrate** in
the representation across the whole dataset. We look at the population, not the individual.

### 5.2 Finding systematic model errors

- **Domino** (Eyuboglu et al., 2022) looks for groups ("slices") of data on which a model does
  badly, by clustering in an embedding space and describing each group in words.

_Difference:_ Domino uses a separate multimodal model and text descriptions. We use a **SOM built
on the classifier's own features**. This gives a fixed 2-D map on which error regions can be seen,
measured, and compared before and after retraining.

### 5.3 Self-Organizing Maps

- **The SOM** (Kohonen, 1990) maps data onto a grid while keeping neighbours close.
- **Clustering the SOM** (Vesanto and Alhoniemi, 2000) shows how a trained map's prototypes can
  be grouped into clusters and evaluated with numbers.
- **SOM-VAE** (Fortuin et al., 2019) and **Deep Embedded SOM** (Forest et al., 2021) train a
  neural network and a SOM **together** to learn better representations.
- **NNSOM** (Jafari) is the open-source SOM library from this program that we build on.

_Difference:_ SOM-VAE and DESOM change how the representation is learned. We keep the CNN
**fixed** and use the SOM only to diagnose it afterwards. That separates the value of the
diagnostic signal from the value of a better representation.

### 5.4 Novelty (out-of-distribution) detection

**Established baselines we will reproduce:**

- **Maximum softmax probability** (Hendrycks and Gimpel, 2017): treat low top-class confidence as a sign of novelty. The simplest baseline.
- **ODIN** (Liang et al., 2018): sharpens that signal with temperature scaling and small input perturbations.
- **Mahalanobis distance** (Lee et al., 2018): measures distance to class averages in feature space.
- **Energy score** (Liu et al., 2020): uses the model's raw output scores instead of probabilities.
- **Deep SVDD** (Ruff et al., 2018): learns a compact one-class representation of normal data.
- **Isolation Forest** (Liu et al., 2008) and **One-Class SVM** (Scholkopf et al., 2001): classical detectors that need no deep model.

**More recent methods we will add for a fair, current comparison:**

- **Deep nearest neighbours** (Sun et al., 2022): score by distance to the nearest training samples in feature space.
- **ViM** (Wang et al., 2022): combines feature-space geometry with the model's output scores.
- **OpenOOD** (Yang et al., 2022): a benchmark and shared protocol for comparing OOD detectors fairly. We follow its idea of a common evaluation setup.

_Difference:_ we do not claim that a SOM beats these methods. We test whether **quantization and
topology signals on fixed CNN embeddings** give a useful novelty signal, and whether the same map
also shows **where** errors concentrate. That combined use is what the other methods do not offer.
A weaker result against strong baselines will be reported as such.

### 5.5 Summary: how this project differs

| Existing work                      | What it gives                            | What we add                                                                          |
| ---------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------ |
| Saliency, LIME, SHAP               | Explanation of one prediction            | A map of **where failures concentrate** over the whole dataset                       |
| Domino and slice discovery         | Groups of hard data, described in text   | A fixed 2-D SOM map, measured and **re-measured after retraining**                   |
| SOM-VAE, DESOM                     | Better representations by joint training | Diagnosis of a **fixed** model, kept separate from representation learning           |
| OOD detectors (MSP to ViM)         | A novelty score                          | The same map gives novelty **and** error geography, compared against those detectors |
| Curriculum and hard-example mining | Retraining schedules                     | Retraining data chosen **from the SOM**, tested against a **random control**         |

---

## 6. Success criteria

The rules below are fixed **before** the test set is examined for each question. This proposal
said the thresholds still open at that time would be filled in here by the **2026-10-06** working session.
Instead they were fixed in the phase protocols, before any test-set use (RQ2:
`ERROR_PATTERN_ROBUSTNESS_CRITERIA.md`, written 2026-10-06; RQ5: `RETRAINING_PROTOCOL.md`, stamped
2026-11-03 for MNIST and 2026-11-22 for Fashion-MNIST). They were copied into this table unchanged on 2026-11-15 (see
`src/v1_mnist/docs/DEVIATIONS.md`). Changing them afterwards would need a written reason.

|         | Counts as supported if                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Counts as not supported if                                                            |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **RQ1** | Descriptive only: report cluster purity, class overlap, quantization and topographic error. No pass/fail.                                                                                                                                                                                                                                                                                                                                                                                                  | n/a                                                                                   |
| **RQ2** | Some cells have an error rate whose Wilson lower bound is above the overall error rate, **and** the top 5% of cells (k = max(5, 5% of neurons), picked by Wilson lower bound on one half of validation) hold more than their share of errors on the other half: the median held-out ratio beats the label-shuffled baseline (p < 0.05) and its 2.5th percentile is above 1 (`ERROR_PATTERN_ROBUSTNESS_CRITERIA.md` §3). No fixed percentage is used, so the rule means the same on every dataset and grid. | Errors are spread across cells about as expected by chance.                           |
| **RQ3** | AUROC for the SOM score is above 0.5 with a bootstrap 95% interval that excludes 0.5, on every ID/OOD pair.                                                                                                                                                                                                                                                                                                                                                                                                | The interval includes 0.5.                                                            |
| **RQ4** | Reported honestly against all baselines. "Competitive" means within 0.01 AUROC of the best baseline: the lower end of the 95% paired bootstrap interval for (SOM AUROC minus best-baseline AUROC) must be above -0.01. Margin fixed on 2026-11-10, after the development results were seen (see `src/v1_mnist/docs/DEVIATIONS.md`). "Better" needs non-overlapping intervals over at least 5 seeds.                                                                                                        | SOM score is clearly worse than simple baselines such as maximum softmax probability. |
| **RQ5** | SOM-guided selection beats the random control on hotspot error rate, as a paired difference over at least 5 seeds with an interval excluding 0, **and** macro-F1 drops by no more than **0.00157** on MNIST and **0.0033** on Fashion-MNIST (2 × the seed-to-seed standard deviation of the `standard` arm's validation macro-F1, measured by the same rule on each dataset before any arms were compared; `RETRAINING_PROTOCOL.md`, stamped 2026-11-03 and 2026-11-22).                                                                                                                            | Guided and random selection are indistinguishable.                                    |
| **RQ6** | Before/after quantization error, topographic error and purity improve in the direction predicted, beyond seed-to-seed variation.                                                                                                                                                                                                                                                                                                                                                                           | No change beyond noise.                                                               |

**A known limit on RQ2.** The MNIST baseline makes 92 mistakes on the 10,500 validation images
and 86 on the 10,500 test images. That is few samples for per-cell statistics. We handle this by
(a) using Wilson bounds, (b) a label-shuffled baseline and held-out hotspots, using validation
images only (the test split is not used for RQ2), and (c) replicating on Fashion-MNIST, where
mistakes are more frequent.

---

## 7. Team and roles

Two people, each owning specific pipeline stages, so every stage has one accountable owner and
becomes one GitHub ticket. Both review each other's pull requests, and both commit every week.

| Stage                                                        | Owner                                      | Reviewer         |
| ------------------------------------------------------------ | ------------------------------------------ | ---------------- |
| Data pipeline and CNN baseline                               | Fyrooz Anika Khan                                | Nazish Atta      |
| Embedding extraction                                         | Fyrooz Anika Khan                                | Nazish Atta      |
| SOM training, grid selection, static and interactive figures | Fyrooz Anika Khan                                | Nazish Atta      |
| Cluster analysis (RQ1)                                       | Nazish Atta, Fyrooz Anika Khan             | Each reviews the other |
| Error-geography and hotspot analysis (RQ2)                   | Nazish Atta, Fyrooz Anika Khan             | Each reviews the other |
| OOD baselines (RQ4)                                          | Nazish Atta                                | Fyrooz Anika Khan      |
| Novelty scoring with SOM (RQ3)                               | Nazish Atta                                | Fyrooz Anika Khan      |
| SOM-guided retraining and random control (RQ5)               | Fyrooz Anika Khan                                | Nazish Atta      |
| Before/after comparison (RQ6)                                | Fyrooz Anika Khan                                | Nazish Atta      |
| LaTeX report and literature review                           | Fyrooz Anika Khan, Nazish Atta                   | The other person |
| Repository maintenance, proposal, DVC                        | Fyrooz Anika Khan                                | Nazish Atta      |
| README, poster, videos                                       | Fyrooz Anika Khan, Nazish Atta                   | The other person |
| Final paper                                                  | Fyrooz Anika Khan, Nazish Atta                   | The other person |

RQ3–RQ6 owners were recorded on 2026-11-15 from who did the work; they were not assigned on
2026-10-06 as planned.

Work moves through pull requests: feature branch, then `develop`, then `main`. `main` always
shows the current state of the project.

---

## 8. Timeline

Class meets on Tuesdays. The semester has 15 weeks. Two class days are lost to breaks (2026-10-13
Fall Break and 2026-11-24 Thanksgiving), and two are the presentations, so **planning is done
in working sessions**.

| Date           | Session                                       | Plan                                                                                                                                                                                                    | Done?     |
| -------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| 2026-09-22     | Solutions and first results                   | MNIST baseline, embeddings, 15x15 SOM, first figures                                                                                                                                                    | Yes       |
| 2026-09-29     | Working session                               | Cluster analysis, error-pattern and hotspot analysis (in review), LaTeX report skeleton                                                                                                                 | Yes       |
| **2026-10-06** | Working session                               | **Proposal in repo. Literature review draft committed. Success-criteria thresholds fixed. Owners set for RQ3 to RQ6. GitHub tickets and milestones created.**                                           | Partly    |
| 2026-10-13     | No class (Fall Break)                         | (Optional buffer for RQ2 write-up)                                                                                                                                                                      | n/a       |
| **2026-10-20** | **Preliminary presentation**                  | RQ1 and RQ2 complete on MNIST. First RQ3 results (SOM novelty score vs. maximum softmax probability). Literature review complete. Slides in `presentation/`. Paper skeleton started. | Yes       |
| 2026-10-27     | Working session                               | RQ4: reproduce the OOD baselines (Isolation Forest, One-Class SVM, autoencoder, MSP, energy, Mahalanobis, nearest-neighbour)                                                                            | Yes       |
| 2026-11-03     | Working session                               | RQ4 complete. Repeat pipeline on Fashion-MNIST.                                                                                                                                                         | Yes, late |
| 2026-11-10     | Working session                               | RQ5: SOM-guided retraining with a random-selection control, at least 5 seeds                                                                                                                            | Yes       |
| 2026-11-17     | Working session                               | RQ6: before/after map metrics. Run the pipeline on Fashion-MNIST.                                                                                                                                       | Yes, late |
| **2026-11-22** | Technical freeze                              | No new experiments. Fashion-MNIST Phases 1 to 4 done. Pull requests opened: Fashion-MNIST into `develop`, then `develop` into `main`. Proposal, progress report and reports updated. | Partly    |
| 2026-11-24     | No class (Thanksgiving)                       | (Buffer)                                                                                                                                                                                                | n/a       |
| 2026-11-26     | Deadline                                      | Every pull request merged or closed; merged branches deleted; paper complete in the journal template on `main`; no open placeholders in this proposal | Planned   |
| 2026-11-29     | Deadline                                      | Paper proof-read by both members; poster draft; progress report up to date; paper and reports build from a clean clone | Planned   |
| 2026-12-01     | Working session                               | Final presentation deck, poster and the script for the 1-minute video; dry run of the talk. (Results were frozen on 2026-11-22.)                                                                                                                    | Planned   |
| **2026-12-08** | **Final presentation and journal submission** | Presentation, journal-style paper submitted, poster, recorded video, and a 1-minute project video                                                                                                       | Planned   |

**Every week:** a short progress report in `reports/Progress_Report/`, and reviewed pull
requests from both team members.

---

## 9. Deliverables and risks

### 9.1 Deliverables

| Deliverable                                                                                     | Due                                   |
| ----------------------------------------------------------------------------------------------- | ------------------------------------- |
| This proposal, in the repository                                                                | 2026-10-06                            |
| Written literature review (10+ cited works, each summarised and positioned) in the LaTeX report | Draft 2026-10-06, complete 2026-10-20 |
| Weekly progress reports                                                                         | Every working session                 |
| Preliminary presentation slides                                                                 | 2026-10-20                            |
| Paper skeleton started                                                                          | 2026-10-20                            |
| Reusable analysis pipeline: data, CNN, embeddings, SOM, analysis, novelty detection, retraining | Final                                 |
| Results tables and vector figures (SVG or PDF), diagrams as draw.io/SVG                         | Final                                 |
| Fashion-MNIST study and the paper merged to `main`; merged branches deleted                     | 2026-11-26                            |
| Target journal and its template named (ticket #40)                                              | 2026-11-26                            |
| Poster draft                                                                                    | 2026-11-29                            |
| Journal-style paper (LaTeX) with bibliography                                                   | 2026-12-08                            |
| Poster, recorded video, 1-minute project video                                                  | 2026-12-08                            |
| Code release: pinned `requirements.txt`, README with install and run steps, tests               | 2026-12-08                            |

### 9.2 Risks and how we handle them

| Risk                                 | Why it matters                                      | What we do                                                                                       |
| ------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Few errors on MNIST                  | 92 validation and 86 test mistakes give weak per-cell statistics | Wilson bounds, label-shuffled baseline, held-out hotspots, and replication on Fashion-MNIST |
| Result only on MNIST                 | Reads as a class project                            | Done: the whole pipeline was repeated on Fashion-MNIST (2026-11-22)                              |
| SOM does not beat strong OOD methods | Possible honest negative result                     | Compare to recent methods; report as-is (it happened on both datasets); the error-geography and retraining findings still stand |
| SOM grid size and training time      | Too small loses detail, too large gives empty cells | Written grid-selection protocol; tune on validation data; use PCA if embeddings are too large    |
| Novelty threshold choice             | A poor threshold distorts results                   | Fit on training-set quantization error only; report AUROC across thresholds                      |
| Retraining is unstable               | Too many hard samples too early can hurt training   | Gradual schedule; monitor validation accuracy; keep the random control                           |
| Only two people                      | Slips affect the whole schedule                     | One owner per stage, weekly reviewed PRs, buffers at 2026-10-13 and 2026-11-24                   |
| Large files bloat the repository     | Git stores every version forever                    | `outputs/**/*.npy` ignored; regenerate or use DVC; keep only small tables in git                 |
| Reproducibility drift                | Numbers that cannot be re-created                   | Fixed seeds, run names that encode configuration, results traced to config                       |

---

## 10. References

_Summaries above are written in our own words. Add each entry below to `references.bib` and
check the details against the original papers when the literature review is finalized._

**Explanation and error discovery**

1. Simonyan, K., Vedaldi, A., Zisserman, A. (2014). Deep inside convolutional networks: Visualising image classification models and saliency maps. _ICLR Workshop_.
2. Ribeiro, M. T., Singh, S., Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. _KDD_.
3. Lundberg, S. M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. _NeurIPS_.
4. Eyuboglu, S., et al. (2022). Domino: Discovering systematic errors with cross-modal embeddings. _ICLR_.

**Self-Organizing Maps**

5. Kohonen, T. (1990). The self-organizing map. _Proceedings of the IEEE_, 78(9).
6. Vesanto, J., Alhoniemi, E. (2000). Clustering of the self-organizing map. _IEEE Transactions on Neural Networks_, 11(3).
7. Fortuin, V., Huser, M., Locatello, F., Strathmann, H., Ratsch, G. (2019). SOM-VAE: Interpretable discrete representation learning on time series. _ICLR_.
8. Forest, F., Lebbah, M., Azzag, H., Lacaille, J. (2021). Deep embedded self-organizing maps for joint representation learning and topology-preserving clustering. _Neural Computing and Applications_, 33.
9. Jafari, A. NNSOM: Neural-network self-organizing map library. https://amir-jafari.github.io/SOM/

**Novelty and out-of-distribution detection**

10. Hendrycks, D., Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. _ICLR_.
11. Liang, S., Li, Y., Srikant, R. (2018). Enhancing the reliability of out-of-distribution image detection in neural networks. _ICLR_.
12. Lee, K., Lee, K., Lee, H., Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. _NeurIPS_.
13. Liu, W., Wang, X., Owens, J., Li, Y. (2020). Energy-based out-of-distribution detection. _NeurIPS_.
14. Ruff, L., et al. (2018). Deep one-class classification. _ICML_.
15. Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). Isolation forest. _ICDM_.
16. Scholkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution. _Neural Computation_, 13(7).
17. Sun, Y., Ming, Y., Zhu, X., Li, Y. (2022). Out-of-distribution detection with deep nearest neighbors. _ICML_.
18. Wang, H., Li, Z., Feng, L., Zhang, W. (2022). ViM: Out-of-distribution with virtual-logit matching. _CVPR_.
19. Yang, J., et al. (2022). OpenOOD: Benchmarking generalized out-of-distribution detection. _NeurIPS Datasets and Benchmarks_.

**Model and datasets**

20. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). Gradient-based learning applied to document recognition. _Proceedings of the IEEE_, 86(11).
21. Xiao, H., Rasul, K., Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747.
22. Netzer, Y., et al. (2011). Reading digits in natural images with unsupervised feature learning. _NIPS Workshop on Deep Learning and Unsupervised Feature Learning_.
