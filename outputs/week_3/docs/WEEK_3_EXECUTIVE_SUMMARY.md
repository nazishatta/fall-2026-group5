# Week 3 Executive Summary: SOM Representation & Manifold Analysis

**Course:** DATS 6501 — Capstone Project  
**Project:** Self-Organizing Maps (NNSOM) on Latent CNN Embeddings  
**Dataset:** MNIST (70,000 samples; 70% Train / 15% Validation / 15% Test)  
**Input Space:** 84-dimensional penultimate layer (`fc2`) embeddings from trained LeNet-5  

---

## 1. Executive Summary & Overview of Week 3 Work

During Week 3, the primary objective was to train, evaluate, and interpret unsupervised **Self-Organizing Maps (SOM)** using the **NNSOM** library on the frozen 84-dimensional feature representations extracted from LeNet-5 (produced in Week 2).

### Key Accomplishments
1. **Unsupervised Pipeline Construction:**
   - Implemented an automated end-to-end SOM pipeline (`04_train_som.py`) with full test-set leakage protection (only `train` embeddings used for fitting/scaling; `test` set strictly held out).
   - Added epoch-by-epoch Quantization Error (QE) trajectory tracking to inspect training convergence across learning phases.
2. **Grid Size Resolution Study:**
   - Trained three comparative grid architectures: **$10\times 10$ (100 neurons)**, **$15\times 15$ (225 neurons)**, and **$20\times 20$ (400 neurons)** using identical seeds and normalization.
   - Performed an objective quantitative comparison based on Quantization Error (QE), 1st-Order Topological Error (TE1), 1st+2nd Order Topological Error (TE1+2), and Neuron Occupancy.
3. **Comprehensive Visualization Suite:**
   - Generated **46 publication-grade figures** (`05_visualize_som.py`) visualizing topological distance (U-Matrix), sample hit histograms, individual class distributions (digits 0–9), top-variance feature histograms, and paginated component planes across all 84 embedding dimensions.

---

## 2. Quantitative Grid Selection Analysis

Before final downstream analysis, three candidate grid architectures were evaluated on the validation set ($N=10,500$):

| Candidate Grid | Total Neurons | Validation QE | QE Rank | Val TE1 (%) | TE1 Rank | Val TE1+2 (%) | TE1+2 Rank | Val Occupancy | Rank Sum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10\times 10$** | 100 | 1.1933 | 3 | **4.88%** | **1** | **2.16%** | **1** | **100.0%** | **5** |
| **$15\times 15$** | 225 | 1.0479 | 2 | 7.52% | 2 | 4.80% | 2 | 99.56% | 6 |
| **$20\times 20$** | 400 | **0.9555** | **1** | 8.96% | 3 | 6.68% | 3 | 99.25% | 7 |

### Scientific Trade-off & Justification for the $20\times 20$ Map:
- **Quantization Error (Reconstruction Fidelity):** The $20\times 20$ grid delivers the lowest Quantization Error ($0.9555$), representing an **$\approx 20\%$ improvement** in vector quantization fidelity over $10\times 10$ ($1.1933$).
- **Resolution for Multi-Class Separation:** In a $10\times 10$ map (100 neurons), each neuron must accommodate roughly $490$ training samples on average across 10 digit classes. The $20\times 20$ map (400 neurons) provides the spatial capacity needed to distinguish fine-grained intra-class styles and isolate confusing inter-class boundaries (e.g., distinguishing 4 vs 9, 3 vs 5, and 7 vs 1).
- **Preserved Topology:** Even with 400 prototype vectors, the $20\times 20$ map maintains over **$99.25\%$ neuron occupancy** on validation data (397 of 400 neurons active) and keeps 1st-order topological error under $9\%$.

---

## 3. Main Findings from Visualizations ($20\times 20$ SOM)

### 3.1. Training Convergence & Learning Dynamics (`convergence/`)
- **Two-Phase Annealing Dynamics:** The Quantization Error curve (`som_20x20_seed42_gridstudy_qe_curve.png`) shows an immediate drop from $\approx 4.35$ down to $\approx 1.10$ within the first 3 epochs (global ordering phase).
- At **Epoch 50**, a distinct step drop occurs (from $1.06$ down to $0.96$), corresponding to the phase transition where neighborhood radius shrinks to fine-tune local topology. The curve stabilizes completely before epoch 100, proving convergence without oscillation.

### 3.2. Manifold Topography & Cluster Boundaries (`core/u_matrix_neuron_distance.png`)
- **Distinct Valley Clusters:** The U-Matrix reveals large, light-yellow contiguous regions ("valleys") representing clusters of high semantic similarity in latent space.
- **Topological Mountain Ridges:** Sharp, dark red/black boundaries ("ridges" of large weight difference) separate distinct basins of attraction. These boundaries visually reflect where the CNN feature extractor creates margins between digit classes.

### 3.3. Hit Density & Neuron Occupancy (`core/train_hit_histogram.png` & `validation_hit_histogram.png`)
- The hit histograms show healthy data utilization across the 400-neuron hexagonal lattice.
- The most populated prototype neurons capture archetypal digit templates ($\approx 40\text{--}69$ validation hits per neuron), while peripheral neurons near the U-matrix boundary ridges have lower counts, acting as smooth interpolators.

### 3.4. Semantic Digit Clustering (`class_maps/`)
By projecting the validation samples of each digit onto the SOM lattice, clear semantic clustering is observed:
- **Digit 0 (`digit_0_color_hist.png`):** Forms a highly concentrated, isolated cluster in the **top-left corner**. Almost no overlap occurs with other digits.
- **Digit 1 (`digit_1_color_hist.png`):** Occupies a compact cluster in the **lower-middle/left region**. Because vertical strokes are visually and structurally distinct, 1 has the sharpest boundaries.
- **Digit 7 (`digit_7_color_hist.png`):** Forms an isolated pocket along the **right perimeter**, positioned close to but clearly separated from digits 4 and 9.
- **Sub-cluster Discovery:** Certain digits exhibit multiple local density peaks within their territory, indicating that the SOM is capturing **distinct writing styles** (e.g., standard straight 7s vs. crossed 7s, loop 2s vs. flat 2s).

### 3.5. Component Plane Analysis (`component_planes/` & `feature_maps/`)
- **Sparse Feature Specialization:** Looking at the paginated component planes (`component_planes_page_01.png` through `page_07.png`), certain LeNet `fc2` features (e.g., `fc2_3`) show uniform activations (inactive/dead units), while features like `fc2_15`, `fc2_19`, and `fc2_79` show strong regional gradients.
- **Orthogonal Concept Encodings:** Top-variance feature planes exhibit complementary directional gradients (e.g., high in top-right vs. high in bottom-left). This confirms that different dimensions of the CNN's penultimate layer encode distinct morphological features that the SOM projects into continuous 2D manifolds.

---

## 4. Methodological Compliance & Scientific Integrity

1. **Strict Data Hygiene:**
   - Preprocessing normalization statistics were derived solely from `train` data.
   - The final `test` split ($10,500$ samples) was completely untouched and held out, ensuring 0% test leakage.
2. **Reproducibility:**
   - All runs were initialized with deterministic seeds (`seed=42`).
   - All 46 generated plots, metrics JSONs, and configuration parameters are tracked with exact filenames and directories.

---

## 5. Next Steps for Week 4
1. **BMU & Receptive Field Extraction:** Complete quantitative Best Matching Unit (BMU) assignment and class purity metrics per neuron (`som_bmu_rq_analysis.py`).
2. **Research Questions (RQ1 & RQ2):**
   - **RQ1 (Representation Quality):** Measure cluster purity and silhouette scores of SOM clusters vs. raw latent space.
   - **RQ2 (Error Geography):** Map CNN misclassifications onto the SOM lattice to determine whether errors concentrate near U-matrix boundary ridges.
