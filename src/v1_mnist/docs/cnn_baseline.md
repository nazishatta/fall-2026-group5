# CNN Baseline – Technical Documentation (v1_mnist)

**Project:** Post-Training Analysis & Novelty Detection Using NNSOM  
**Course:** GWU DATS 6501 Capstone | Advisor: Dr. Amir Jafari  
**Phase:** CNN Baseline – Data Preparation, LeNet-5 Training, Embedding Extraction

---

## Pipeline Overview

The CNN baseline consists of three sequential pipeline scripts, each driven by a single
YAML config file (`src/v1_mnist/component/configs/cnn_baseline.yaml`). All outputs land in
`outputs/v1_mnist/cnn_baseline/` automatically via the config system.

```
Raw MNIST IDX files (~/Capstone/All_Data/MNIST/raw/)
         |
         v
 [01_prepare_data.py]
   - Combines original 60k train + 10k test -> 70k total
   - Stratified 70/15/15 split  -> train=49,000 / val=10,500 / test=10,500
   - Saves split statistics JSON + class-distribution figure
         |
         v
 [02_train_baseline.py]
   - Trains LeNet-5 for 20 epochs (Adam, lr=0.001, StepLR gamma=0.1 @ step 10)
   - Evaluates on val set each epoch; saves lenet5_best.pth and lenet5_final.pth
   - Saves training-curve figures and test_evaluation.json
         |
         v
 [03_extract_embeddings.py]
   - Loads lenet5_best.pth
   - Registers a forward hook on the fc2 layer
   - Runs train / val / test sets through the frozen model
   - Saves {split}_features.npy  (shape: N x 84)
         |
         v
 outputs/v1_mnist/cnn_baseline/embeddings/  <-- ready for NNSOM training
```

---

## Why LeNet-5?

| Criterion               | Justification                                                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Architecture match**  | LeNet-5 was designed for 28x28 greyscale handwritten-digit images — precisely the MNIST format. No resizing needed.                              |
| **Interpretability**    | Small, well-understood layers make it straightforward to justify each design decision to a professor or reviewer.                                |
| **Feature quality**     | The fc2 layer (84-dim) is a well-established feature representation for MNIST; prior NNSOM literature (e.g., Jafari et al.) uses this embedding. |
| **Compute efficiency**  | ~60k parameters — trains to >99% accuracy on a CPU in ~5 min; a GPU cuts that to ~90 sec.                                                        |
| **Pedagogical purpose** | As the _baseline_ model, it must be simple enough that any performance gap opened by the SOM analysis in later weeks is meaningful.              |

### Reference Materials

- LeCun, Y. et al. (1998). "Gradient-Based Learning Applied to Document Recognition." _Proceedings of the IEEE_, 86(11), 2278–2324.
- Jafari, A. et al. NNSOM: Neural-Network Self-Organizing Maps (course material, GWU DATS 6501).
- PyTorch documentation: `torch.nn`, `DataLoader`, forward hooks.

---

## Why fc2 (84-dim)?

LeNet-5's penultimate layer `fc2` is the last hidden representation before the
classification head (`fc3`). It has the following properties:

- **Discriminative**: it has already been transformed by two convolutional blocks and `fc1`
  (120-dim), so digit-class structure is strongly encoded.
- **Compact**: 84 dimensions strike a balance between information density and the
  memory/compute overhead of NNSOM training.
- **Standard**: the 84-dim fc2 embedding is the de-facto MNIST feature vector used in
  post-training analysis research.

Extracting it with a _forward hook_ (rather than modifying `forward()`) means the model
code is not touched during inference — a clean separation of concerns.

---

## Performance Metrics

| Metric                   | Value                                    |
| ------------------------ | ---------------------------------------- |
| Test Accuracy            | **99.18%**                               |
| Macro F1-Score           | **0.9918**                               |
| Best Validation Accuracy | ~99.20% (epoch ~15)                      |
| Easiest digit            | 0 (F1 = 0.9976, 1 error / 1,035 samples) |
| Hardest digits           | 8 (F1 = 0.9878), 6 (F1 = 0.9889)         |

**Why these metrics?**

- _Accuracy_ is the primary MNIST benchmark metric for comparability with the literature.
- _Macro F1_ catches per-class imbalance that accuracy can mask.
- _Confusion matrix_ (saved to `figures/`) reveals which digit pairs the network confuses,
  motivating the NNSOM post-training analysis.

---

## What the Embeddings Are Used For (NNSOM Training)

The 84-dimensional fc2 vectors from `outputs/v1_mnist/cnn_baseline/embeddings/` are the direct input to the
Neural-Network Self-Organizing Map (NNSOM). The SOM will:

1. Cluster the 49,000 training embeddings into a 2-D topographic grid.
2. Map each test embedding to its Best Matching Unit (BMU).
3. Visualise which SOM nodes correspond to which digit classes — revealing internal
   representation geometry and potential novelty/anomaly regions.
4. Identify digits where the SOM neighbourhood is mixed (e.g., 8 and 3), confirming
   the confusion patterns already visible in the LeNet-5 confusion matrix.

---

## Config Architecture

All hyperparameters live in modular YAML files:

| File                                         | Role                                                   |
| -------------------------------------------- | ------------------------------------------------------ |
| `src/v1_mnist/component/configs/base.yaml`   | Shared defaults (model, dataset, training, dataloader) |
| `src/v1_mnist/component/configs/cnn_baseline.yaml` | CNN baseline overrides (`run.name: cnn_baseline`) |
| `src/v1_mnist/component/configs/som.yaml`    | SOM configuration (`run.name: som`)                   |

`config.py` auto-merges these and auto-routes all output paths to `outputs/v1_mnist/{run.name}/**`.

---

_Generated: 2026-09-13 | Author: Capstone Group 5_
