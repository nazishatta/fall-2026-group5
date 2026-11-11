"""Novelty benchmark: SVHN comparison against the required baselines.

Steps (scripts in p3_novelty_detection/novelty_benchmark/), in the order they run:
  1. svhn_benchmark: five-method MNIST validation versus SVHN comparison.
  2. historical_baseline_verification: recompute the earlier MNIST-only summaries.
  3. required_baselines: max-softmax, energy and Mahalanobis on the frozen CNN.
  4. required_baselines_knn: the same baselines plus kNN (k=5).
  5. qe_msp_paired_comparison: paired AUROC comparison of QE with max-softmax.
  6. svhn_score_export: save the energy and kNN score arrays (needed by step 7).
  7. qe_energy_knn_paired: paired AUROC comparisons of QE with energy and kNN.
  8. qe_umatrix_random_ood: paired QE and U-Matrix comparison with a random-score control.
  9. standardized_autoencoder: autoencoder with train-scaled inputs.
 10. fair_budget_baselines: baselines fitted on the same training rows, five seeds.
 11. unified_novelty_benchmark: all nine detectors in one run.
 12. svhn_inverted_norm_diagnostic: embedding norms of SVHN versus inverted SVHN.
 13. qe_norm_dependence: how closely QE follows the embedding norm, and QE's AUROC among
     images with ordinary norms (checks that QE still gives the recorded AUROC).
 14. autoencoder_scaling_diagnostic: autoencoder and Isolation Forest AUROC by training
     size and input scaling, with a length-only control score (reads step 11's results).
 15. missed_svhn_figure: figure of the SVHN images the QE detector misses.
 16. historical_svhn_inspection: print the SVHN score separation of the earlier run
     (read-only, writes nothing).
 17. consolidate_evidence: join the headline numbers of three saved runs into one table.

Results go to outputs/v1_mnist/p3_novelty_detection/novelty_benchmark/<run_id>/.
Only MNIST validation and SVHN are scored; the held-out MNIST test split is never used.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p3_novelty_detection/run_novelty_benchmark.py \
      [--only STEP [--config CONFIG]] [--dry-run]

Every step has its own config in src/v1_mnist/component/configs/ (shown by --dry-run).
Needs the frozen CNN checkpoint and embeddings (Phase 1), the trained SOM (Phase 2)
and the SVHN test file (DVC). A step whose results folder already exists is skipped
(results are never overwritten); to repeat a step, give it a new run_id in a new
config and run it with --only STEP --config.

Not a step, because its command line differs (it takes explicit file arguments):
  - run_ood_comparison: compares two score files (needs --id-scores, --ood-scores,
    --output, --run-id, --detector and --config).
Run it with python -m src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.<name>.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.v1_mnist.component.pipeline.p3_novelty_detection.phase3_steps import P3Step, run_phase3  # noqa: E402

FOLDER = "novelty_benchmark"
STEPS = [
    P3Step("svhn_benchmark", "Five-method comparison: MNIST validation versus SVHN",
           f"{FOLDER}/run_svhn_benchmark.py", "novelty_benchmark.yaml"),
    P3Step("historical_baseline_verification", "Recompute the earlier MNIST-only baseline summaries",
           f"{FOLDER}/run_historical_baseline_verification.py", "historical_baseline_verification.yaml", ("--save",)),
    P3Step("required_baselines", "Max-softmax, energy and Mahalanobis on the frozen CNN",
           f"{FOLDER}/run_required_baselines.py", "required_ood_baselines.yaml", ("--save",)),
    P3Step("required_baselines_knn", "Required baselines plus kNN",
           f"{FOLDER}/run_required_baselines.py", "svhn_knn_baselines.yaml", ("--save",)),
    P3Step("qe_msp_paired_comparison", "Paired AUROC: QE versus max-softmax",
           f"{FOLDER}/run_required_baselines.py", "svhn_paired_comparison.yaml", ("--save",)),
    P3Step("svhn_score_export", "Save the energy and kNN score arrays",
           f"{FOLDER}/run_required_baselines.py", "svhn_score_export.yaml", ("--save",)),
    P3Step("qe_energy_knn_paired", "Paired AUROC: QE versus energy and kNN",
           f"{FOLDER}/run_qe_energy_knn_paired.py", "qe_energy_knn_paired.yaml"),
    P3Step("qe_umatrix_random_ood", "Paired QE and U-Matrix comparison with a random-score control",
           f"{FOLDER}/run_qe_umatrix_random_ood.py", "qe_u_matrix_random_ood.yaml"),
    P3Step("standardized_autoencoder", "Autoencoder with train-scaled inputs",
           f"{FOLDER}/run_standardized_autoencoder.py", "autoencoder_standardized_diagnostic.yaml"),
    P3Step("fair_budget_baselines", "Baselines fitted on the same training rows, five seeds",
           f"{FOLDER}/run_fair_budget_baselines.py", "fair_budget_baselines.yaml"),
    P3Step("unified_novelty_benchmark", "All nine detectors in one run",
           f"{FOLDER}/unified_novelty_benchmark.py", "unified_novelty_benchmark.yaml"),
    P3Step("svhn_inverted_norm_diagnostic", "Embedding norms: SVHN versus inverted SVHN",
           f"{FOLDER}/svhn_inverted_norm_diagnostic.py", "svhn_inverted_norm_diagnostic.yaml"),
    P3Step("qe_norm_dependence", "QE versus embedding norm: correlation and a norm-matched AUROC",
           f"{FOLDER}/run_qe_norm_dependence.py", "qe_norm_dependence.yaml"),
    P3Step("autoencoder_scaling_diagnostic", "Autoencoder and Isolation Forest by training size and scaling",
           f"{FOLDER}/run_autoencoder_scaling.py", "autoencoder_scaling_diagnostic.yaml"),
    P3Step("missed_svhn_figure", "Figure: SVHN images the QE detector misses",
           f"{FOLDER}/run_missed_svhn_figure.py", "missed_svhn_figure.yaml"),
    P3Step("historical_svhn_inspection", "Print the SVHN score separation of the earlier run (read-only)",
           f"{FOLDER}/inspect_svhn_scores.py", "historical_svhn_inspection.yaml"),
    P3Step("consolidate_evidence", "Join the headline numbers of three saved runs into one table",
           f"{FOLDER}/consolidate_phase3_evidence.py", "phase3_evidence_consolidation.yaml"),
]


def main() -> None:
    run_phase3("Novelty benchmark", "SVHN novelty benchmark against the required baselines.", STEPS)


if __name__ == "__main__":
    main()
