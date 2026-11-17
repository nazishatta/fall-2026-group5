# Weekly Progress Report

## Date: Week 1 - August 25, 2026

- Topics of discussion
  - Selected the NNSOM post-training analysis project proposal.
  - Reviewed and understood the overall project workflow.
  - Selected MNIST and LeNet-5 for the initial baseline experiment.

- Action Items:

* [x] Review and select project proposal
* [x] Understand project objectives and workflow
* [x] Select MNIST dataset and LeNet-5 baseline

---

## Date: Week 2 - September 1, 2026

- Topics of discussion
  - **PHASE 1: MODEL TRAINING BASELINE**
  - Set up Python environment with PyTorch, NNSOM, matplotlib, scikit-learn, and required packages.
  - Set up AWS EC2 connection.
  - Set up the GitHub repository, its architecture and configuration files, using the modular `src/v1_mnist/component/` structure.
  - Built the MNIST data pipeline with a 70/15/15 stratified train/validation/test split, and the LeNet-5 model components.
  - Trained the LeNet-5 baseline CNN and achieved approximately **99.18% test accuracy**.
  - Extracted and saved the **84-dimensional `fc2` embeddings** for train, validation, and test sets.
  - Saved checkpoints, embeddings, training curves, and confusion matrix under `outputs/v1_mnist/p1_model_training_baseline/cnn_baseline_training/`.
  - Studied Chapter 16 on Competitive Networks to understand SOM, BMU, competitive learning, neighborhood updates, and topology preservation.

- Action Items:

* [x] Set up project environment and AWS EC2
* [x] Set up GitHub workflow and project structure
* [x] Prepare MNIST data pipeline
* [x] Train and evaluate LeNet-5 baseline CNN
* [x] Extract and save CNN embeddings

---

## Date: Week 3 - September 8, 2026

- Topics of discussion
  - **PHASE 1: MODEL TRAINING BASELINE**
  - Recorded the baseline evaluation: test metrics, class counts of each split, and a provenance manifest and validation report for the embeddings.
  - **PHASE 2: SOM TRAINING & VISUALIZATION**
  - Set up and explored NNSOM using the saved CNN embeddings.
  - Configured the SOM hyperparameters: grid size, starting neighbourhood and number of epochs.
  - Built the SOM training pipeline: trained the SOM on the training embeddings and monitored the quantization error.
  - Made the initialization memory-safe (economy SVD) and saved every SOM run under its own run ID.
  - Visualized the trained SOM using component planes, hit histogram, and U-Matrix.

- Action Items:

* [x] Finalize NNSOM configuration
* [x] Train SOM on CNN embeddings
* [x] Monitor quantization error
* [x] Generate component planes, hit histogram, and U-Matrix

---

## Date: Week 4 - September 15, 2026

- Topics of discussion
  - **PHASE 2: SOM TRAINING & VISUALIZATION**
  - Wrote and froze the grid-selection protocol and ran it on the 10x10, 15x15 and 20x20 grids.
  - Assigned the samples to their Best Matching Units and located the class regions and the misclassified samples on the map.
  - Added a check that stops a run from overwriting an earlier run.
  - Reorganized the repository into the versioned `src/v1_mnist/` structure.
  - Selected the 15x15 grid and re-plotted the SOM figures.
  - Tracked the CNN checkpoints, embeddings and SOM model with DVC.

- Action Items:

* [x] Grid-selection protocol
* [x] Map samples to BMUs
* [x] Reorganize git based on versioning
* [x] Select 15x15 grid and re-plot the figures
* [x] Track checkpoints, embeddings and the SOM model with DVC

---

## Date: Week 5 - September 22, 2026

- Topics of discussion
  - **PHASE 2: SOM TRAINING & VISUALIZATION**
  - Tuned the 15x15 SOM (starting neighbourhood 11, 250 epochs) and regenerated the component planes, hit histogram and U-Matrix.
  - Added interactive SOM plots: quantization error across epochs, neighbourhood radius, and train/validation quality.
  - Drew the project pipeline diagram (draw.io).
  - Wrote the SOM visualization report.

- Action Items:

* [x] Tune the 15x15 SOM and update its figures
* [x] Interactive SOM visualizations
* [x] Project pipeline diagram

---

## Date: Week 6 - September 29, 2026

- Topics of discussion
  - **PHASE 2: SOM TRAINING & VISUALIZATION**
  - Cluster analysis: assigned the samples to their Best Matching Units, color-coded the SOM by true class, and computed cluster purity and class-cluster intersection.
  - Error pattern analysis: separated correct and misclassified samples on the map, identified error hotspots, and retrieved the hard samples in them.
  - Wrote the group proposal: research questions, success criteria, related work, plan and roles.
  - Set up shared DVC storage on S3, pinned the package versions, and wrote the LaTeX reports for the CNN baseline, SOM visualization, cluster analysis and error patterns.

- Action Items:

* [x] Cluster analysis
* [x] Error pattern analysis
* [x] Group proposal
* [x] DVC shared storage

---

## Date: Week 7 - October 6, 2026

- Topics of discussion
  - **PHASE 2: SOM TRAINING & VISUALIZATION**
  - Compared the 15x15 and 20x20 grids with a pre-registered, equal-budget test: neither grid predicts the CNN's mistakes better.
  - Checked the error-hotspot result on all three grids against a label-shuffled baseline: errors gather in hotspots on every grid.
  - Organized the code and outputs by phase (p1 to p4) and trained the final SOMs with model checks and a run manifest.
  - **PHASE 3: NOVELTY DETECTION**
  - Scored each sample by its quantization error, with a threshold fitted on the training-set quantization error (95th percentile).
  - Computed the U-Matrix and a U-Matrix-based novelty score, and compared it with quantization-error scoring.
  - Benchmarked SOM novelty detection against Isolation Forest, One-Class SVM and an autoencoder.

- Action Items:

* [x] Grid comparison
* [x] Error hotspot robustness check
* [x] Quantization-error novelty detection
* [x] U-Matrix novelty scoring
* [x] Classical novelty baselines

---

## Date: Week 8 - October 13, 2026

- Topics of discussion
  - **PHASE 2: SOM TRAINING & VISUALIZATION**
  - Compared SOM error hotspots with k-means cells, kNN impurity and the CNN's maximum softmax probability.
  - **PHASE 3: NOVELTY DETECTION**
  - Evaluated novelty detection with SVHN as out-of-distribution data (AUROC, AUPR, FPR at 95% TPR), against maximum softmax probability, energy and Mahalanobis scores.
  - Wrote the Phase 3 report, the first final report, and the preliminary presentation slides.

- Action Items:

* [x] Error-pattern baselines
* [x] SVHN novelty benchmark
* [x] Phase 3 report
* [x] Preliminary presentation slides

---

## Date: Week 9 - October 20, 2026

- Topics of discussion
  - Preliminary Presentation.
  - **PHASE 3: NOVELTY DETECTION**
  - Compared nine novelty methods on the same samples, with multiple seeds.
  - Analyzed failure modes: SVHN embeddings are much shorter than MNIST embeddings, which explains part of the separation.
  - Drew the novelty detection framework diagram.

- Action Items:

* [x] Preliminary presentation
* [x] Nine-method novelty benchmark
* [x] Failure-mode analysis
* [x] Novelty framework diagram

---

## Date: Week 10 - October 27, 2026

- Topics of discussion
  - **PHASE 3: NOVELTY DETECTION**
  - Organized the novelty detection code into step folders and connected it to the pipeline, with tests.
  - **PHASE 4: GUIDED RETRAINING**
  - Designed the retraining: a hard-example pool from SOM error hotspots and low-purity cells, a curriculum, targeted augmentation, random and max-softmax controls, and 5 seeds.

- Action Items:

* [x] Phase 3 pipeline integration
* [x] Phase 4 design

---

## Date: Week 11 - November 3, 2026

- Topics of discussion
  - **PHASE 4: GUIDED RETRAINING**
  - Hard example mining: built the hard-example pool and compared curriculum training with standard training.
  - Targeted data augmentation: retrained with augmented copies of the hotspot samples and measured the change in hotspot error and elsewhere (CF1 and CF2).
  - SOM-guided augmentation lowered the hotspot test error from 11.4% to 7.8% without harming the rest of the data; curriculum training did not help.
  - Iterative SOM re-analysis: re-extracted features from the retrained models, re-trained the SOM, and compared the old and new maps.
  - Wrote the Phase 4 report.

- Action Items:

* [x] Hard example mining and curriculum comparison
* [x] Targeted data augmentation
* [x] SOM re-analysis
* [x] Phase 4 report

---

## Date: Week 12 - November 10, 2026

- Topics of discussion
  - **PHASE 5: PAPER WRITING & CODE RELEASE**
  - Results consolidation: cleaned the Phase 3 runs, added a run manifest, and updated the final report with Phases 3 and 4.
  - Tuned the SOM settings of all three grids with Optuna and checked the chosen settings on 5 new seeds: the 15x15 setting used in all results is confirmed.
  - Counted the test errors fixed by retraining and the error per digit.
  - Added automatic testing (CI) and updated the proposal with the results so far.
  - Started the research paper in LaTeX and selected the target journal (*Neurocomputing*).

- Action Items:

* [x] Results consolidation
* [x] Optuna tuning
* [x] Errors fixed and per-digit error
* [x] Research paper started

---

## Date: Week 13 - November 17, 2026

- Topics of discussion
  - **PHASE 3: NOVELTY DETECTION**
  - Near-novelty test with held-out classes: a LeNet-5 and a SOM are trained on 6 digits and the other 4 digits are treated as novel, over 5 class splits (the standard open-set setup). Wrote the protocol before any run, built the code and tests, and ran it on EC2. The SOM detects the held-out digits in every split (mean AUROC 0.912), but plain nearest-neighbour distance does better in every split (0.966) and the comparison with maximum softmax probability (0.944) is inconclusive.
  - **PHASE 4: GUIDED RETRAINING**
  - Novelty detection after retraining: scored the 15 retrained models with quantization error, maximum softmax probability, energy and Mahalanobis on MNIST against SVHN. The SOM still detects SVHN after every kind of retraining (lowest AUROC 0.965); SOM-guided retraining did not change it clearly, and maximum softmax probability and energy remain better.
  - **PHASE 5: PAPER WRITING & CODE RELEASE**
  - Completed the final report: a figure explaining how the SOM works (with the Competitive Networks chapter cited), the contribution statement in the abstract, the Optuna and novelty-after-retraining results, all diagrams redrawn in draw.io, and clearer result tables.
  - Same-budget comparison of the error detectors: flagging the same 1.6% of validation images as the SOM's hotspot cells, the least-confident images hold 53.3% of the CNN's mistakes and kNN impurity 48.7%, against 18.1% in the hotspot cells (exploratory).
  - Plan for the rest of the MNIST work: re-analyse the SOM of the least-confident retraining arm (running on EC2).
  - Plan after that: run Phases 2 to 4 on Fashion-MNIST (technical freeze on November 22), merge develop into main through a reviewed pull request, and write the research paper with one results table per dataset.

- Action Items:

* [x] Novelty detection after retraining
* [x] Final report: SOM figure, contribution statement, draw.io diagrams, citations
* [x] Held-out-class novelty: protocol, code and tests
* [x] Held-out-class novelty results
* [ ] SOM re-analysis of the least-confident retraining arm
* [x] Same-budget comparison of the error detectors
* [ ] Merge develop into main
* [ ] Fashion-MNIST pipeline and results
* [ ] Research paper draft

---

## Date: Week 14 - November 24, 2026 (planned)

- Topics of discussion
  - **PHASE 5: PAPER WRITING & CODE RELEASE**
  - Fashion-MNIST results through Phases 2 to 4, using the same code, SOM settings and statistics as MNIST.
  - Research paper draft with the MNIST and Fashion-MNIST results.
  - Prepare the poster (due November 29) and the final presentation (due December 1); recorded videos and the journal submission follow by December 8.

- Action Items:

* [ ] Fashion-MNIST results
* [ ] Research paper draft
* [ ] Poster
* [ ] Final presentation
