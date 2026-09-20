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
  - Set up GitHub repository and organized the project using the professor's `src/component/` structure.
  - Prepared MNIST with a 70/15/15 stratified train/validation/test split.
  - Trained the LeNet-5 baseline CNN and achieved approximately **99.18% test accuracy**.
  - Extracted and saved the **84-dimensional `fc2` embeddings** for train, validation, and test sets.
  - Saved checkpoints, embeddings, training curves, and confusion matrix under `outputs/week_2/`.
  - Studied Chapter 16 on Competitive Networks to understand SOM, BMU, competitive learning, neighborhood updates, and topology preservation.

- Action Items:

* [x] Set up project environment and AWS EC2
* [x] Set up GitHub workflow and project structure
* [x] Prepare MNIST data pipeline
* [x] Train and evaluate LeNet-5 baseline CNN
* [x] Extract and save CNN embeddings

---

## Date: Week 3 - September 7, 2026

- Topics of discussion
  - **PHASE 2: SOM TRAINING & VISUALIZATION**
  - Set up and explore NNSOM using the saved CNN embeddings.
  - Configure SOM hyperparameters: grid size, learning rate, sigma, and number of epochs.
  - Train SOM on the training embeddings and monitor quantization error.
  - Visualize the trained SOM using component planes, hit histogram, and U-Matrix.
  - Assign train/test samples to Best Matching Units (BMUs) and analyze class cluster regions.
  - Compute cluster purity and class-cluster intersection metrics.
  - Create confusion heatmaps on the SOM to identify class overlap and confusion regions.

- Action Items:

* [ ] Finalize NNSOM configuration
* [ ] Train SOM on CNN embeddings
* [ ] Monitor quantization error
* [ ] Generate component planes, hit histogram, and U-Matrix
* [ ] Map samples to BMUs and color-code by true class
* [ ] Compute cluster purity and class-cluster intersection metrics
* [ ] Create SOM confusion heatmaps

---
