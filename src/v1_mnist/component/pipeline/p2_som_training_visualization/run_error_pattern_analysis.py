"""Error pattern analysis and its robustness check.

Steps:
  1. error_patterns: per-neuron error geography, hotspots (the project's one
     hotspot rule, utils/hotspots.py), confusion pairs and the list of
     misclassified validation samples.
  2. hotspot_samples: images of the actual MNIST digits in the hotspots.
  3. error_figures: error-rate and complex error maps.
  4. error_heatmaps: error heatmaps (error count, rate, Wilson bound, confidence, hotspots).
  5. error_pattern_robustness: error-pattern statistics for the headline SOM and each robustness SOM.
  6. error_pattern_robustness_verdict: apply the pre-specified rules to those results.

Results go to outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_error_pattern_analysis.py \
      [--config CONFIG] [--robustness-configs CFG ...] [--mnist-root PATH] [--dry-run]

Needs the cluster analysis results for the same SOM (run run_cluster_analysis.py
first). Stops if this SOM's error-pattern results already exist (results are never overwritten).
Robustness results that already exist are kept and their steps skipped.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.v1_mnist.component.pipeline.common import (  # noqa: E402
    P2, REPO_ROOT, SOM_CONFIG, Step, log, base_parser, check_not_done, load, resolve, run_steps,
)


def main() -> None:
    parser = base_parser("Error pattern analysis and robustness check.", SOM_CONFIG)
    parser.add_argument("--robustness-configs", nargs="*", default=None,
                        help="Configs of the robustness SOMs (default: robustness_check.configs in --config).")
    parser.add_argument("--mnist-root", default=None,
                        help="Folder holding the torchvision MNIST download (passed to hotspot_samples.py).")
    args = parser.parse_args()

    config = load(args.config)
    model = str(config.visualization.selected_model)
    if args.robustness_configs is None:
        args.robustness_configs = [str(c) for c in config.robustness_check.configs]
    robustness_models = [str(load(c).visualization.selected_model) for c in args.robustness_configs]
    cluster_summary = resolve(config.paths.cluster_dir) / model / "cluster_summary.json"
    error_dir = resolve(config.paths.error_dir)

    if not cluster_summary.exists() and not args.dry_run:
        log.error(f"Cluster analysis results are missing ({cluster_summary.relative_to(REPO_ROOT).as_posix()}).\n"
              "Run run_cluster_analysis.py with the same config first.")
        sys.exit(1)
    check_not_done([error_dir / model / "error_summary.json"], args.dry_run,
                   hint=f"Folder: {error_dir / model}")

    folder = P2 / "error_pattern_analysis"
    hotspot_args = ["--config", args.config] + (["--mnist-root", args.mnist_root] if args.mnist_root else [])
    steps = [
        Step("Error geography, hotspots, confusion pairs", folder / "error_patterns.py", ["--config", args.config]),
        Step("Images of the digits in the hotspots", folder / "hotspot_samples.py", hotspot_args),
        Step("Error-rate and complex error maps",
             folder / "error_figures.py", ["--config", args.config, "--model-name", model]),
        Step("Error heatmaps", folder / "error_heatmaps.py", ["--config", args.config]),
    ]
    # Robustness results are never overwritten: existing ones are kept and reused.
    robustness_dir = error_dir / "robustness"
    for cfg, m in zip([args.config, *args.robustness_configs], [model, *robustness_models]):
        if (robustness_dir / f"{m}_error_pattern_robustness.json").exists():
            log.info(f"Robustness result for {m} already exists; kept (step skipped).")
            continue
        steps.append(Step(f"Error-pattern robustness statistics: {Path(cfg).stem}",
                          folder / "error_pattern_robustness.py", ["--config", cfg]))
    if (robustness_dir / "error_pattern_robustness_verdict.json").exists():
        log.info("Robustness verdict already exists; kept (step skipped).")
    else:
        steps.append(Step("Error-pattern robustness verdict", folder / "error_pattern_robustness_verdict.py",
                          ["--config", args.config, "--headline", model]
                          + (["--robustness", *robustness_models] if robustness_models else [])))
    run_steps(f"Error pattern analysis ({model})", steps, args.dry_run)


if __name__ == "__main__":
    main()
