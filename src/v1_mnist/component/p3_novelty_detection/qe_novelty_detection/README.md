# Novelty detection with quantization error (`src/v1_mnist/component/p3_novelty_detection/qe_novelty_detection/`)

Proposal step: Novelty Detection via Quantization Error. Uses the SOM's quantization error (QE, the distance from an image's embedding to its nearest neuron) as a novelty score. The cut-off is set from the training images only and then frozen. The score is tested on MNIST validation images against SVHN images (house numbers, the outside dataset), with a cut-off sweep and a random-selection control.

Runner: `pipeline/p3_novelty_detection/run_qe_novelty_detection.py`. Results: `outputs/v1_mnist/p3_novelty_detection/qe_novelty_detection/` (one folder per run ID). Notes: `docs/p3_novelty_detection/qe_novelty_detection/`.

## Files

| File | What it is |
|---|---|
| `novelty_figures.py` | Small PDF-only figures for training-calibrated QE validation analysis. |
| `ood_adapters.py` | Dataset-specific image adapters; imported lazily to keep numeric tests lightweight. |
| `ood_metrics.py` | Dataset-independent OOD metrics; higher scores always indicate novelty. |
| `ood_scoring.py` | Score frozen CNN embeddings using the upstream SOM BMU-distance convention. |
| `qe_threshold_sweep.py` | Threshold sweep for the QE novelty score: operating points and a threshold-free AUROC. |
| `quantization_novelty.py` | Training-calibrated SOM BMU-distance novelty scoring, with no test access. |
| `run_qe_novelty.py` ▶ | Run Phase 3 QE baseline against frozen, hash-verified Phase 2 artifacts. |
| `run_qe_ood.py` ▶ | Independent development-only MNIST-to-OOD QE evaluation; no MNIST test access. |
| `run_qe_random_control.py` ▶ | Recompute validation-only QE enrichment against uniform random selection. |
| `run_qe_threshold_sweep.py` ▶ | Report the QE novelty score at several percentile cutoffs, as a new run. |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
