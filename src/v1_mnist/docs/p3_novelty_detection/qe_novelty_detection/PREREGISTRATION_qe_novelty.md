# Preregistration: Quantization Error Novelty Detection

Written: 2026-10-11. **Status: proposed development protocol; official OOD results not examined.**

## 1. Question
Can frozen SOM quantization-error (QE) scores distinguish MNIST from SVHN and Fashion-MNIST? Report the two ID/OOD pairs separately.

## 2. Prior observations
The existing MNIST 15x15 SOM (`som_15x15_nb11_ep250_s42_v2`) uses LeNet-5 `fc2` (84 features). Training QE's 95th percentile is **1.5690279077294162**. Among 10,500 MNIST validation images, 539 exceed that threshold; 22 of 92 CNN errors are flagged. These are known development observations, **not OOD detection rates**. No external OOD results have been measured in this work.

## 3. Frozen inputs and leakage control
Use the existing MNIST CNN checkpoint, SOM weights, and training-fitted MinMaxScaler [-1,1] unchanged. Reconstruct the scaler using **MNIST training embeddings only**. No OOD or final-test data may fit a model, scaler, or threshold. Use the pre-existing strict decision rule `QE > 1.5690279077294162`; do not change the distance definition. Do not use `load_som_data()` during development: it automatically opens MNIST test embeddings.

## 4. Evaluation inputs
ID development reference: all 10,500 existing MNIST validation samples. Evaluate **SVHN official test** and **Fashion-MNIST official test** separately, in their entirety, without selection based on performance. Fashion-MNIST: grayscale 28x28, scale to [0,1]. SVHN: RGB-to-grayscale with Torchvision `Grayscale(num_output_channels=1)`, then `Resize((28,28), interpolation=BILINEAR, antialias=True)`, then `ToTensor()`; pin/record the torchvision version and exact transformation behavior. No OOD fitting or tuning.

## 5. Endpoints and statistical decision
Primary: QE AUROC (OOD=1, ID=0, larger score more novel) for each pair. The project's prespecified success rule is AUROC > 0.5 with a 95% bootstrap confidence interval excluding 0.5 **on every pair**. Report negative/inconclusive outcomes unchanged. Secondary: OOD-positive average precision (AUPR), FPR@95TPR (lowest FPR at observed operating points attaining TPR >=0.95), OOD detection rate and ID false-positive rate at the frozen threshold, and sample counts. AUPR depends on class prevalence: report counts.

Confidence interval: stratified bootstrap (independently resample ID and OOD with replacement), 2,000 iterations, `numpy.random.default_rng(42)`, percentile 2.5/97.5 bounds. Fixed before evaluation. Results from this validation-reference design are **development results**, not final held-out MNIST test claims.

## 6. Baselines and controls
On **identical evaluation samples**, compare QE with MSP (`1 - max softmax`), energy (fix temperature before evaluation), and Mahalanobis (train-only class statistics). The professor's separate benchmark objective additionally requires Isolation Forest, One-Class SVM, and autoencoder reconstruction error. No baseline numbers are claimed until measured. Use agreed budgets/seeds and keep exploratory controls separate from the main comparisons.

## 7. Reproducibility and reporting
Record source SHA-256 checksums, preprocessing versions, OOD split/counts, metric conventions, chosen seeds, and a unique run ID. Refuse existing output paths; never overwrite prior results. Small metrics go in JSON/CSV. Figures, if needed, are PDF only. The same metrics/scoring functions may be reused with future datasets but every new **ID dataset** requires its own trained CNN, SOM, scaler, and training-calibrated QE threshold. Final MNIST test evaluation requires a separate, frozen final-evaluation specification and reconciliation of any applicable unresolved proposal success criteria.
