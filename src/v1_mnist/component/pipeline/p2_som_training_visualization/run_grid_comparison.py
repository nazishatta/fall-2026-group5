"""Pre-registered SOM grid comparison (Phase 2): do the 10x10, 15x15 and 20x20 maps differ in how well
their cells predict held-out CNN mistakes?

Rules: src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/PREREGISTRATION_grid_comparison.md

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_grid_comparison.py --stage all [--config CONFIG]

Stages, in order (``all`` runs them in this order): ``tune`` (Stage 1: three neighbourhoods per grid, full
training set), ``choose`` (one neighbourhood per grid from validation QE, TE1 and TE1+2; labels not used),
``boot`` (Stage 2: 10 maps per grid on the same bootstrap resamples), ``analyze`` (the endpoint over 200
repeats, results.json), ``figure`` (PDF). A finished map is skipped, so ``tune`` and ``boot`` can be re-run
after an interruption. For MNIST, ``verify_saved`` checks the port against the original run's saved maps
(exact), and ``all`` ends with ``verify_retrain`` (the fresh run against the original, within the config's
tolerance). Only training and validation data are read; the test split is never loaded.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.settings import (  # noqa: E402
    DEFAULT_CONFIG,
    load_grid_settings,
)
from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402

LOGGER = logging.getLogger("v1_mnist.grid_comparison")
STAGES = ("tune", "choose", "boot", "analyze", "figure")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=(*STAGES, "all", "verify_saved", "verify_retrain"), required=True)
    args = parser.parse_args(argv)
    settings = load_grid_settings(args.config, REPO_ROOT)
    setup_logger("v1_mnist", settings.output_dir / "logs" / f"{args.stage}.log")
    LOGGER.info("Run %s, stage %s", settings.run_id, args.stage)

    from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison import (
        analysis,
        figure,
        provenance,
        stages,
    )

    if args.stage == "verify_saved":
        analysis.verify_saved_maps(settings, provenance.load_data(settings))
        return
    if args.stage == "verify_retrain":
        provenance.verify_run(settings)
        analysis.verify_retrain(settings, json.loads(settings.results_path.read_text(encoding="utf-8")))
        return
    todo = STAGES if args.stage == "all" else (args.stage,)
    if "tune" in todo:
        provenance.start_run(settings)
    else:
        provenance.verify_run(settings)
    data = provenance.load_data(settings)
    result = None
    for stage in todo:
        LOGGER.info("Stage %s", stage)
        if stage == "tune":
            stages.run_tune(settings, data)
        elif stage == "choose":
            stages.run_choose(settings)
        elif stage == "boot":
            stages.run_boot(settings, data)
        elif stage == "analyze":
            result = analysis.run_analyze(settings, data, stages.load_choice(settings))
        else:
            figure.run_figure(settings)
    if args.stage == "all" and settings.reproduction is not None and result is not None:
        analysis.verify_retrain(settings, result)


if __name__ == "__main__":
    main()
