# Capstone Proposal: Where Do Neural Networks Fail?

## Post-Training Analysis and Novelty Detection Using Neural-Network Self-Organizing Maps (NNSOM)

| | |
|---|---|
| **Course** | DATS 6501 Capstone, Fall 2026, The George Washington University |
| **Group** | Group 5 |
| **Team** | Fyrooz Khan, Nazish Atta |
| **Advisor** | Dr. Amir Jafari |
| **Repository** | https://github.com/nazishatta/fall-2026-group5 |
| **Version** | Group proposal, revised 2026-09-30 |

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

### Where we are today (2026-09-30)

- LeNet-5 baseline on MNIST: **99.18% test accuracy**, macro-F1 0.9918.
- 84-dimensional embeddings extracted from the CNN's second fully connected layer (fc2).
- A 15x15 SOM trained on those embeddings (run `som_15x15_seed42_final`), with static and
  interactive figures.
- Cluster analysis and error-pattern analysis code, and a first LaTeX report with a related-work
  section, are written and under review.

---

## 2. Research questions

| | Question | Main output |
|---|---|---|
| **RQ1** | What does the CNN's feature space look like on a topology-preserving map? | Cluster structure, class overlap, density maps |
| **RQ2** | Do misclassifications sit in identifiable regions of that map? | Error-hotspot statistics and example images |
| **RQ3** | Can SOM-derived scores identify out-of-distribution (OOD) samples? | AUROC, AUPR, FPR at 95% TPR |
| **RQ4** | How does SOM novelty detection compare with established methods? | Side-by-side table against baselines (Section 5) |
| **RQ5** | Can SOM error regions pick samples worth retraining on? | SOM-guided selection vs. a **random-selection control** |
| **RQ6** | After retraining, is the representation measurably better organized? | Before/after map metrics |

---

## 3. Data

### 3.1 Datasets and access terms

All datasets are public. None contains personal data. All download automatically through
`torchvision`, so no accounts or approvals are needed. Raw data is never committed to git.

| Dataset | Role | Size | Access and terms |
|---|---|---|---|
| **MNIST** | Main development dataset | 70,000 grayscale 28x28 digit images, 10 classes | `torchvision.datasets.MNIST`. Originally distributed from Yann LeCun's site and widely used for research. **We have not re-checked its exact terms; confirm them on that page.** Cite LeCun et al. (1998). |
| **Fashion-MNIST** | Second dataset (replication), and OOD set for MNIST | 70,000 grayscale 28x28 images, 10 clothing classes | `torchvision.datasets.FashionMNIST`. **MIT licence.** Cite Xiao et al. (2017). |
| **CIFAR-10** | Harder dataset for the final results | 60,000 colour 32x32 images, 10 classes (50,000 train, 10,000 test) | `torchvision.datasets.CIFAR10`. No formal licence stated; the authors require citing Krizhevsky (2009). |
| **SVHN** | OOD set (street-view house numbers) | 73,257 train and 26,032 test digit crops | `torchvision.datasets.SVHN`. **For non-commercial use only**, which fits this academic project. Cite Netzer et al. (2011). |
| **EuroSAT** (stretch goal) | A more realistic, non-digit domain (satellite land-use images) | 27,000 labelled images, 10 classes | Public on GitHub, **MIT licence**. Cite Helber et al. (2019). |

Licence and terms for Fashion-MNIST, CIFAR-10, SVHN and EuroSAT were checked against each
dataset's own page on 2026-09-30. MNIST's terms still need checking (see the table). Please
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
(the S3 storage for DVC is not accessible yet). Small metrics tables (`.csv`, `.json`) **are**
committed, because those are what a reader needs to check our numbers.

### 3.4 Datasets beyond MNIST

MNIST is good for building the method, but a reviewer will ask whether it says anything about
models that matter. We therefore commit to **Fashion-MNIST and CIFAR-10 for the final
results**, and will decide by 2026-10-20 whether to add EuroSAT as a more realistic example.

---

## 4. Method (the pipeline)

```
data -> CNN baseline -> embeddings (fc2) -> SOM (NNSOM) -> analysis -> novelty detection -> guided retraining
```

The diagram is in the README (`demo/fig/project_pipeline/nnsom_project_pipeline.drawio.svg`).

1. **Baseline.** Train LeNet-5 (MNIST, Fashion-MNIST) and ResNet-18 (CIFAR-10) with fixed seeds.
   Report accuracy, per-class F1, and the confusion matrix.
2. **Embeddings.** Freeze the trained model. Save the penultimate-layer vector for every sample
   in every split, with sample IDs, labels, predictions and confidence, so results can be traced.
3. **SOM.** Train a 15x15 SOM on training-set embeddings with NNSOM. Choose the grid size using
   the written protocol (`src/v1_mnist/docs/som/SOM_GRID_SELECTION_PROTOCOL.md`) and
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
config to raw result to table or figure. Unmeasured values are written as `[TBD]`, never guessed.

---

## 5. Literature review and state of the art

We group related work into four themes. For each we say what it does, in plain words, and how
this project differs. Full references are in Section 10.

### 5.1 Explaining a model one prediction at a time

- **Saliency maps** (Simonyan et al., 2014) colour the pixels that most change a prediction.
- **LIME** (Ribeiro et al., 2016) fits a small, simple model around one prediction to explain it.
- **SHAP** (Lundberg and Lee, 2017) assigns each input feature a fair share of credit for one
  prediction, using ideas from game theory.

*Difference:* all three explain a single sample. None shows where failures **concentrate** in
the representation across the whole dataset. We look at the population, not the individual.

### 5.2 Finding systematic model errors

- **Domino** (Eyuboglu et al., 2022) looks for groups ("slices") of data on which a model does
  badly, by clustering in an embedding space and describing each group in words.

*Difference:* Domino uses a separate multimodal model and text descriptions. We use a **SOM built
on the classifier's own features**. This gives a fixed 2-D map on which error regions can be seen,
measured, and compared before and after retraining.

### 5.3 Self-Organizing Maps

- **The SOM** (Kohonen, 1990) maps data onto a grid while keeping neighbours close.
- **Clustering the SOM** (Vesanto and Alhoniemi, 2000) shows how a trained map's prototypes can
  be grouped into clusters and evaluated with numbers.
- **SOM-VAE** (Fortuin et al., 2019) and **Deep Embedded SOM** (Forest et al., 2021) train a
  neural network and a SOM **together** to learn better representations.
- **NNSOM** (Jafari) is the open-source SOM library from this program that we build on.

*Difference:* SOM-VAE and DESOM change how the representation is learned. We keep the CNN
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

*Difference:* we do not claim that a SOM beats these methods. We test whether **quantization and
topology signals on fixed CNN embeddings** give a useful novelty signal, and whether the same map
also shows **where** errors concentrate. That combined use is what the other methods do not offer.
A weaker result against strong baselines will be reported as such.

### 5.5 Summary: how this project differs

| Existing work | What it gives | What we add |
|---|---|---|
| Saliency, LIME, SHAP | Explanation of one prediction | A map of **where failures concentrate** over the whole dataset |
| Domino and slice discovery | Groups of hard data, described in text | A fixed 2-D SOM map, measured and **re-measured after retraining** |
| SOM-VAE, DESOM | Better representations by joint training | Diagnosis of a **fixed** model, kept separate from representation learning |
| OOD detectors (MSP to ViM) | A novelty score | The same map gives novelty **and** error geography, compared against those detectors |
| Curriculum and hard-example mining | Retraining schedules | Retraining data chosen **from the SOM**, tested against a **random control** |

---

## 6. Success criteria

The rules below are fixed **before** the test set is examined for each question. Numeric
thresholds marked `[TBD]` will be filled in and committed by the **2026-10-06** working session,
before any RQ3 to RQ6 test-set evaluation. Changing them afterwards would need a written reason.

| | Counts as supported if | Counts as not supported if |
|---|---|---|
| **RQ1** | Descriptive only: report cluster purity, class overlap, quantization and topographic error. No pass/fail. | n/a |
| **RQ2** | Some cells have an error rate whose Wilson lower bound is above the overall error rate, **and** a small share of cells holds `[TBD]`% of all errors, **and** a label-shuffling test shows this is unlikely by chance. | Errors are spread across cells about as expected by chance. |
| **RQ3** | AUROC for the SOM score is above 0.5 with a bootstrap 95% interval that excludes 0.5, on every ID/OOD pair. | The interval includes 0.5. |
| **RQ4** | Reported honestly against all baselines. "Competitive" means within `[TBD]` AUROC of the best baseline across seeds. "Better" needs non-overlapping intervals over at least 5 seeds. | SOM score is clearly worse than simple baselines such as maximum softmax probability. |
| **RQ5** | SOM-guided selection beats the random control on hotspot error rate, as a paired difference over at least 5 seeds with an interval excluding 0, **and** macro-F1 drops by no more than `[TBD]`. | Guided and random selection are indistinguishable. |
| **RQ6** | Before/after quantization error, topographic error and purity improve in the direction predicted, beyond seed-to-seed variation. | No change beyond noise. |

**A known limit on RQ2.** The MNIST baseline makes about 86 mistakes on the 10,500 test images
(0.82% error). That is few samples for per-cell statistics. We handle this by (a) using Wilson
bounds, (b) pooling validation and test errors for analysis only, not for tuning, and
(c) replicating on Fashion-MNIST and CIFAR-10, where errors are more frequent.

---

## 7. Team and roles

Two people, each owning specific pipeline stages, so every stage has one accountable owner and
becomes one GitHub ticket. Both review each other's pull requests, and both commit every week.

| Stage | Owner | Reviewer |
|---|---|---|
| Data pipeline and CNN baseline | Fyrooz Khan | Nazish Atta |
| Embedding extraction | Fyrooz Khan | Nazish Atta |
| SOM training, grid selection, static and interactive figures | Fyrooz Khan | Nazish Atta |
| Cluster analysis (RQ1) | Nazish Atta | Fyrooz Khan |
| Error-geography and hotspot analysis (RQ2) | Nazish Atta | Fyrooz Khan |
| OOD baselines (RQ4) | `[TBD, decide 2026-10-06]` | The other person |
| Novelty scoring with SOM (RQ3) | `[TBD, decide 2026-10-06]` | The other person |
| SOM-guided retraining and random control (RQ5) | `[TBD]` | The other person |
| Before/after comparison (RQ6) | `[TBD]` | The other person |
| LaTeX report and literature review | Nazish Atta (lead), Fyrooz Khan (co-write) | The other person |
| Proposal, README, repository hygiene, DVC | Fyrooz Khan | Nazish Atta |
| Final paper, poster, videos | Both | Both |

Work moves through pull requests: feature branch, then `develop`, then `main`. `main` always
shows the current state of the project.

---

## 8. Timeline

Class meets on Tuesdays. The semester has 15 weeks. Two class days are lost to breaks (2026-10-13
Fall Break and 2026-11-24 Thanksgiving), and two are the presentations, so **planning is done
in working sessions**.

| Date | Session | Plan | Done? |
|---|---|---|---|
| 2026-09-22 | Solutions and first results | MNIST baseline, embeddings, 15x15 SOM, first figures | Yes |
| 2026-09-29 | Working session | Cluster analysis, error-pattern and hotspot analysis (in review), LaTeX report skeleton | In review |
| **2026-10-06** | Working session | **Proposal in repo. Literature review draft committed. Success-criteria thresholds fixed. Owners set for RQ3 to RQ6. GitHub tickets and milestones created.** | Planned |
| 2026-10-13 | No class (Fall Break) | (Optional buffer for RQ2 write-up) | n/a |
| **2026-10-20** | **Preliminary presentation** | RQ1 and RQ2 complete on MNIST. First RQ3 results (SOM novelty score vs. maximum softmax probability). Literature review complete. Slides in `presentation/`. Paper skeleton started. Decide on EuroSAT. | Planned |
| 2026-10-27 | Working session | RQ4: reproduce the OOD baselines (Isolation Forest, One-Class SVM, autoencoder, MSP, energy, Mahalanobis, nearest-neighbour) | Planned |
| 2026-11-03 | Working session | RQ4 complete. Repeat pipeline on Fashion-MNIST. | Planned |
| 2026-11-10 | Working session | RQ5: SOM-guided retraining with a random-selection control, at least 5 seeds | Planned |
| 2026-11-17 | Working session | RQ6: before/after map metrics. Run the pipeline on CIFAR-10. | Planned |
| 2026-11-24 | No class (Thanksgiving) | (Buffer) | n/a |
| 2026-12-01 | Working session | Freeze results. Full paper draft. Poster. Videos recorded. Code cleaned for release. | Planned |
| **2026-12-08** | **Final presentation and journal submission** | Presentation, journal-style paper submitted, poster, recorded video, and a 1-minute project video | Planned |

**Every week:** a short progress report in `reports/Progress_Report/`, and reviewed pull
requests from both team members.

---

## 9. Deliverables and risks

### 9.1 Deliverables

| Deliverable | Due |
|---|---|
| This proposal, in the repository | 2026-10-06 |
| Written literature review (10+ cited works, each summarised and positioned) in the LaTeX report | Draft 2026-10-06, complete 2026-10-20 |
| Weekly progress reports | Every working session |
| Preliminary presentation slides | 2026-10-20 |
| Paper skeleton started | 2026-10-20 |
| Reusable analysis pipeline: data, CNN, embeddings, SOM, analysis, novelty detection, retraining | Final |
| Results tables and vector figures (SVG or PDF), diagrams as draw.io/SVG | Final |
| Journal-style paper (LaTeX) with bibliography | 2026-12-08 |
| Poster, recorded video, 1-minute project video | 2026-12-08 |
| Code release: pinned `requirements.txt`, README with install and run steps, tests | 2026-12-08 |

### 9.2 Risks and how we handle them

| Risk | Why it matters | What we do |
|---|---|---|
| Few errors on MNIST | About 86 test errors gives weak per-cell statistics | Wilson bounds, pooled analysis (not for tuning), and replication on Fashion-MNIST and CIFAR-10 |
| Result only on MNIST | Reads as a class project | Commit to Fashion-MNIST and CIFAR-10 for final results; consider EuroSAT |
| SOM does not beat strong OOD methods | Possible honest negative result | Compare to recent methods; report as-is; the error-geography and retraining findings still stand |
| SOM grid size and training time | Too small loses detail, too large gives empty cells | Written grid-selection protocol; tune on validation data; use PCA if embeddings are too large |
| Novelty threshold choice | A poor threshold distorts results | Fit on training-set quantization error only; report AUROC across thresholds |
| Retraining is unstable | Too many hard samples too early can hurt training | Gradual schedule; monitor validation accuracy; keep the random control |
| Only two people | Slips affect the whole schedule | One owner per stage, weekly reviewed PRs, buffers at 2026-10-13 and 2026-11-24 |
| Large files bloat the repository | Git stores every version forever | `outputs/**/*.npy` ignored; regenerate or use DVC; keep only small tables in git |
| DVC storage (S3) not yet reachable | Cannot share large arrays | Regenerate from code with fixed seeds; add DVC pointers once access works |
| Reproducibility drift | Numbers that cannot be re-created | Fixed seeds, run names that encode configuration, results traced to config |

---

## 10. References

*Summaries above are written in our own words. Add each entry below to `references.bib` and
check the details against the original papers when the literature review is finalized.*

**Explanation and error discovery**

1. Simonyan, K., Vedaldi, A., Zisserman, A. (2014). Deep inside convolutional networks: Visualising image classification models and saliency maps. *ICLR Workshop*.
2. Ribeiro, M. T., Singh, S., Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD*.
3. Lundberg, S. M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
4. Eyuboglu, S., et al. (2022). Domino: Discovering systematic errors with cross-modal embeddings. *ICLR*.

**Self-Organizing Maps**

5. Kohonen, T. (1990). The self-organizing map. *Proceedings of the IEEE*, 78(9).
6. Vesanto, J., Alhoniemi, E. (2000). Clustering of the self-organizing map. *IEEE Transactions on Neural Networks*, 11(3).
7. Fortuin, V., Huser, M., Locatello, F., Strathmann, H., Ratsch, G. (2019). SOM-VAE: Interpretable discrete representation learning on time series. *ICLR*.
8. Forest, F., Lebbah, M., Azzag, H., Lacaille, J. (2021). Deep embedded self-organizing maps for joint representation learning and topology-preserving clustering. *Neural Computing and Applications*, 33.
9. Jafari, A. NNSOM: Neural-network self-organizing map library. https://amir-jafari.github.io/SOM/

**Novelty and out-of-distribution detection**

10. Hendrycks, D., Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR*.
11. Liang, S., Li, Y., Srikant, R. (2018). Enhancing the reliability of out-of-distribution image detection in neural networks. *ICLR*.
12. Lee, K., Lee, K., Lee, H., Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *NeurIPS*.
13. Liu, W., Wang, X., Owens, J., Li, Y. (2020). Energy-based out-of-distribution detection. *NeurIPS*.
14. Ruff, L., et al. (2018). Deep one-class classification. *ICML*.
15. Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). Isolation forest. *ICDM*.
16. Scholkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution. *Neural Computation*, 13(7).
17. Sun, Y., Ming, Y., Zhu, X., Li, Y. (2022). Out-of-distribution detection with deep nearest neighbors. *ICML*.
18. Wang, H., Li, Z., Feng, L., Zhang, W. (2022). ViM: Out-of-distribution with virtual-logit matching. *CVPR*.
19. Yang, J., et al. (2022). OpenOOD: Benchmarking generalized out-of-distribution detection. *NeurIPS Datasets and Benchmarks*.

**Model and datasets**

20. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11).
21. Xiao, H., Rasul, K., Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747.
22. Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.
23. Netzer, Y., et al. (2011). Reading digits in natural images with unsupervised feature learning. *NIPS Workshop on Deep Learning and Unsupervised Feature Learning*.
24. Helber, P., et al. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*.
