"""Command line shared by the retraining runners (curriculum comparison, targeted augmentation).

Stages (run them in this order):
  train        train every arm and seed that is not finished yet (validation images only)
  summary      validation summary of all runs; prints the suggested macro-F1 guard
  final_test   the single test pass; refuses until the protocol is stamped
Options: --arm NAME / --seed N to run part of the work, --smoke for a quick 2-epoch, 1-seed trial
that writes to a separate "_smoke" folder (delete it afterwards).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.settings import (
    DEFAULT_CONFIG,
)
from src.v1_mnist.component.utils.logging import setup_logger
from src.v1_mnist.component.utils.retraining_data import load_retraining_data
from src.v1_mnist.component.utils.retraining_final import run_final_test
from src.v1_mnist.component.utils.retraining_plans import BuildArm
from src.v1_mnist.component.utils.retraining_runs import run_arms
from src.v1_mnist.component.utils.retraining_settings import (
    ExperimentSettings,
    load_experiment_settings,
)
from src.v1_mnist.component.utils.retraining_stability import (
    plot_validation_curves,
    stability_summary,
)
from src.v1_mnist.component.utils.retraining_summary import write_validation_summary
from src.v1_mnist.component.utils.run_provenance import write_json

LOGGER = logging.getLogger("v1_mnist.retraining_cli")
STAGES = ("train", "summary", "final_test")


def smoke_settings(settings: ExperimentSettings) -> ExperimentSettings:
    """A throw-away copy: 2 epochs, the first seed, its own folder."""
    return dataclasses.replace(
        settings,
        run_id=f"{settings.run_id}_smoke",
        output_dir=settings.output_dir.with_name(f"{settings.output_dir.name}_smoke"),
        seeds=settings.seeds[:1],
        train=dataclasses.replace(settings.train, epochs=2),
    )


def run_experiment_cli(
    section: str,
    description: str,
    make_builder: Callable[[ExperimentSettings], BuildArm],
    repo_root: Path,
    argv: list[str] | None = None,
) -> None:
    """Parse the command line and run the chosen stage of one experiment."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--arm", action="append", help="Only this arm (repeat for several).")
    parser.add_argument("--seed", type=int, action="append", help="Only this seed (repeat for several).")
    parser.add_argument("--smoke", action="store_true", help="2 epochs, first seed only, separate folder.")
    args = parser.parse_args(argv)

    settings = load_experiment_settings(section, args.config, repo_root)
    if args.smoke:
        if args.stage == "final_test":
            sys.exit("--smoke never touches the test split.")
        settings = smoke_settings(settings)
    setup_logger("v1_mnist", settings.output_dir / "logs" / f"{args.stage}.log")
    LOGGER.info("Run %s, stage %s", settings.run_id, args.stage)

    if args.stage == "train":
        data = load_retraining_data(settings, repo_root)
        done = run_arms(settings, data, make_builder(settings), args.arm, args.seed)
        LOGGER.info("%d run(s) finished now.", done)
    elif args.stage == "summary":
        summary = write_validation_summary(settings)
        stability = stability_summary(settings)
        write_json(settings.tables_dir / "training_stability.json", stability)
        plot_validation_curves(settings, settings.output_dir / "figures" / "validation_accuracy_by_epoch")
        for arm_name, numbers in stability.items():
            LOGGER.info("%s: largest one-epoch fall in validation accuracy %.4f (outside hotspots %.4f)",
                        arm_name, numbers["largest_drop_overall"]["mean"],
                        numbers["largest_drop_non_hotspot"]["mean"])
        LOGGER.info("Macro-F1 guard suggestion (2 x standard-arm std): %s", summary.get("macro_f1_guard_suggestion"))
    else:
        run_final_test(settings, repo_root)
