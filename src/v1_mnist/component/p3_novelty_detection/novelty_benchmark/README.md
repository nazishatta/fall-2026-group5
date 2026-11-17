# Novelty benchmark (`src/v1_mnist/component/p3_novelty_detection/novelty_benchmark/`)

Proposal Week 8. Compares the SOM scores (QE, U-Matrix) with other novelty detectors on the same embeddings: Isolation Forest, One-Class SVM, an autoencoder, max-softmax probability, energy, Mahalanobis distance and k-nearest-neighbour distance. Reports AUROC, AUPR and FPR at 95% TPR, with paired bootstrap intervals. Also holds the diagnostic checks (for example whether QE just follows the length of the embedding) and the script that joins all Phase 3 evidence.

Runner: `pipeline/p3_novelty_detection/run_novelty_benchmark.py`. Results: `outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/`. Notes: `docs/p3_novelty_detection/novelty_benchmark/`.

## Files

| File | What it is |
|---|---|
| `autoencoder_baseline.py` | Feature-reconstruction autoencoder for MNIST novelty benchmarking. |
| `autoencoder_scaling.py` | Controlled check of what drives the autoencoder and Isolation Forest AUROC on SVHN. |
| `classical_baselines.py` | MNIST classical novelty baselines on CNN embeddings. |
| `cnn_score_inference.py` | Frozen CNN score reconstruction and validation for Phase 3. |
| `consolidate_phase3_evidence.py` ▶ | Build a traceable Phase 3 comparison from immutable EC2 result JSONs. |
| `inspect_svhn_scores.py` ▶ | Inspect historical SVHN novelty-score separation without refitting or altering it. |
| `knn_baseline.py` | k-nearest-neighbor distance baseline for novelty detection. |
| `ood_comparison.py` | Held-out OOD comparison; never fit detectors or calibrate on evaluation samples. |
| `paired_bootstrap.py` | Paired stratified bootstrap for comparing two OOD AUROCs. |
| `paired_ood_comparison.py` | Frozen-score paired OOD comparison for Phase 3. |
| `qe_norm_dependence.py` | How much of the QE novelty signal is an embedding-norm effect? |
| `required_baselines.py` | Required novelty baselines on frozen CNN outputs and embeddings. |
| `run_autoencoder_scaling.py` ▶ | Controlled autoencoder / Isolation Forest check on SVHN, as a brand-new run. |
| `run_fair_budget_baselines.py` ▶ | Post-hoc equal-training-pool novelty baseline diagnostic. |
| `run_historical_baseline_verification.py` ▶ | Recompute historical MNIST-only validation summaries in a NEW run folder. |
| `run_missed_svhn_figure.py` ▶ | Reconstruct frozen QE missed-SVHN examples from historical IDs. |
| `run_qe_energy_knn_paired.py` ▶ | Paired AUROC comparisons of SOM QE against energy and kNN. |
| `run_qe_norm_dependence.py` ▶ | Is part of the QE novelty result an embedding-norm effect? |
| `run_qe_umatrix_random_ood.py` ▶ | Reproduce Phase 3 QE/U-Matrix paired and random-score OOD analyses. |
| `run_required_baselines.py` ▶ | Frozen-CNN max-softmax, energy, Mahalanobis and optional kNN evaluation. |
| `run_standardized_autoencoder.py` ▶ | Additive Phase 3 diagnostic: train-scaled autoencoder on frozen CNN embeddings. |
| `run_svhn_benchmark.py` ▶ | Independent SVHN comparison using frozen MNIST models and baseline settings. |
| `svhn_inverted_norm_diagnostic.py` ▶ | Frozen-CNN SVHN versus intensity-inverted SVHN norm diagnostic. |
| `unified_novelty_benchmark.py` ▶ | Single-run, nine-detector exploratory MNIST-validation versus SVHN benchmark. |
| `unified_novelty_metrics.py` | Evaluation and uncertainty for the exploratory unified novelty benchmark. |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
