# Phase 3: Novelty Detection (`src/v1_mnist/component/p3_novelty_detection/`)

Tests whether the SOM can tell unfamiliar images from familiar ones, using the frozen Phase 1 CNN and the official SOM. MNIST validation images are the familiar set and SVHN is the outside set; the held-out test split is not used. **Order:** `qe_novelty_detection` → `u_matrix_analysis` → `novelty_benchmark`, then the extra `held_out_class_novelty` test. The newest run folder for each result is listed in the "Start here" table of `outputs/v1_mnist/p3_novelty_detection/README.md`.

## Folders

| Folder | What is inside |
|---|---|
| [`qe_novelty_detection/`](qe_novelty_detection/README.md) | Proposal Week 6: uses the SOM's quantization error (QE, the distance from an image's embedding to its nearest neuron) as a novelty score. |
| [`u_matrix_analysis/`](u_matrix_analysis/README.md) | Proposal Week 7: the U-Matrix shows how far each neuron is from its neighbours, so it marks borders between regions of the map. |
| [`novelty_benchmark/`](novelty_benchmark/README.md) | Proposal Week 8: compares the SOM scores (QE, U-Matrix) with other novelty detectors on the same embeddings: Isolation Forest, One-Class SVM, an autoencoder, max-softmax probability, energy, Mahalanobis distance and k-nearest-neighbour distance. |
| [`held_out_class_novelty/`](held_out_class_novelty/README.md) | A harder ("near") novelty test, asked for in Review #5. |
