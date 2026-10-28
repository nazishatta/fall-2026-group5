"""Adapter connecting Phase 3 scripts to the shared pipeline interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Allow direct execution from the shared pipeline.
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger("v1_mnist.pipeline.phase3")

PIPELINE_DIR = Path(__file__).resolve().parent

STAGES = {
    "qe_novelty": (
        "run_qe_novelty.py",
        ["--save"],
    ),
    "qe_ood": (
        "run_qe_ood.py",
        ["--evaluate", "--save"],
    ),
    "svhn_benchmark": (
        "run_svhn_benchmark.py",
        [],
    ),
    "historical_baseline_verification": (
        "run_historical_baseline_verification.py",
        ["--save"],
    ),
    "qe_energy_knn_paired": (
        "run_qe_energy_knn_paired.py",
        [],
    ),
    "qe_random_control": (
        "run_qe_random_control.py",
        ["--save"],
    ),
    "qe_u_matrix": (
        "run_qe_u_matrix.py",
        ["--save"],
    ),
    "qe_umatrix_random_ood": (
        "run_qe_umatrix_random_ood.py",
        [],
    ),
    "required_baselines": (
        "run_required_baselines.py",
        ["--save"],
    ),
    "standardized_autoencoder": (
        "run_standardized_autoencoder.py",
        [],
    ),
    "qe_umatrix_validation_figure": (
        "run_qe_umatrix_validation_figure.py",
        [],
    ),
    "u_matrix_grid_figure": (
        "run_u_matrix_grid_figure.py",
        [],
    ),
    "missed_svhn_figure": (
        "run_missed_svhn_figure.py",
        [],
    ),
    "ood_comparison": (
        "run_ood_comparison.py",
        [],
    ),
    "proposal_gap_analysis": (
        "run_proposal_gap_analysis.py",
        [],
    ),
    "fair_budget_baselines": (
        "run_fair_budget_baselines.py",
        [],
    ),
}

BENCHMARK_CONFIG = (
    REPO_ROOT
    / "src/v1_mnist/component/configs/novelty_benchmark.yaml"
)


def build_command(
    stage: str,
    config: Path,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build one Phase 3 stage command without executing it."""
    script, extra = STAGES[stage]
    resolved_config = (
        config if config.is_absolute() else REPO_ROOT / config
    ).resolve()

    if stage == "svhn_benchmark":
        # Historical benchmark currently reads this fixed YAML location.
        if resolved_config != BENCHMARK_CONFIG.resolve():
            raise ValueError(
                "SVHN benchmark currently requires novelty_benchmark.yaml"
            )
        return [sys.executable, str(PIPELINE_DIR / script)]

    if stage == "ood_comparison":
        if not extra_args:
            raise ValueError(
                "OOD comparison requires --id-scores, --ood-scores, "
                "--output, --run-id and --detector."
            )
        return [
            sys.executable,
            str(PIPELINE_DIR / script),
            "--config",
            str(resolved_config),
            *extra_args,
        ]

    return [
        sys.executable,
        str(PIPELINE_DIR / script),
        "--config",
        str(resolved_config),
        *extra,
    ]


def main() -> None:
    """Execute or preview a configured Phase 3 stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args()
    if forwarded and args.stage != "ood_comparison":
        parser.error("Additional arguments are only allowed for ood_comparison")
    if args.stage == "ood_comparison":
        required = {
            "--id-scores", "--ood-scores", "--output",
            "--run-id", "--detector",
        }
        if not required.issubset(set(forwarded)):
            parser.error("OOD comparison requires all five named arguments")

    command = build_command(args.stage, args.config, forwarded)
    LOGGER.info("Phase 3 stage: %s", args.stage)
    LOGGER.info("Command: %s", " ".join(command))

    if args.dry_run:
        LOGGER.info("Dry run: no experiment executed")
        return

    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
