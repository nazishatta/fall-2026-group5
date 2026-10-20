"""Non-destructive review-readiness audit for MNIST Phase 3 PR #17."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)

REQUIRED_METRICS = {
    "mnist_svhn_five_method_comparison_s42_initial": "comparison_metrics.json",
    "mnist_svhn_required_baselines_knn_s42_v1": "required_baseline_metrics.json",
    "mnist_svhn_ae_train_standardized_s42_v1": "standardized_autoencoder_metrics.json",
}
FORBIDDEN_TRACKED_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".png", ".jpg", ".jpeg", ".svg"}


def git(root: Path, *args: str) -> str:
    """Run a read-only git command."""
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if result.returncode != 0 and args[:1] != ("config",):
        raise RuntimeError(result.stderr.strip() or f"Git failed: {args}")
    return result.stdout


def audit(root: Path) -> tuple[list[str], list[str]]:
    """Return blockers and advisories without modifying the repository."""
    blockers: list[str] = []
    advisories: list[str] = []
    expected = root / "src/v1_mnist/component/pipeline/p3_novelty_detection/run_svhn_benchmark.py"
    if not expected.is_file():
        blockers.append("Not the expected Phase 3 repository root")
        return blockers, advisories

    for filename in ("common.py", "run_all.py"):
        path = f"src/v1_mnist/component/pipeline/{filename}"
        if git(root, "diff", "eb3d837^", "HEAD", "--", path).strip():
            blockers.append(f"Historical shared pipeline commit changed: {path}")
        if git(root, "diff", "--", path).strip():
            blockers.append(f"Uncommitted shared pipeline change: {path}")

    for run, filename in REQUIRED_METRICS.items():
        path = root / "outputs/v1_mnist/p3_novelty_detection/novelty_benchmark" / run / filename
        if not path.is_file():
            blockers.append(f"Missing MNIST-SVHN measured evidence: {path.relative_to(root)}")
            continue
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(content, dict):
                blockers.append(f"Metrics JSON is not an object: {path.relative_to(root)}")
        except (ValueError, OSError) as error:
            blockers.append(f"Invalid result JSON: {path.relative_to(root)}: {error}")

    for relative in (
        "src/v1_mnist/component/configs/autoencoder_standardized_diagnostic.yaml",
        "src/v1_mnist/component/pipeline/p3_novelty_detection/run_standardized_autoencoder.py",
        "src/v1_mnist/docs/DEVIATIONS.md",
    ):
        if not (root / relative).is_file():
            blockers.append(f"Missing essential file: {relative}")
    deviations = (root / "src/v1_mnist/docs/DEVIATIONS.md").read_text(encoding="utf-8")
    if "Phase 3 — novelty threshold and OOD benchmark" not in deviations:
        blockers.append("Training-vs-validation novelty-threshold deviation undocumented")

    for path in git(root, "ls-files", "--cached").splitlines():
        suffix = Path(path).suffix.lower()
        if suffix in FORBIDDEN_TRACKED_SUFFIXES and not path.endswith(".drawio.svg"):
            advisories.append(f"Review tracked binary/figure against PDF-only policy: {path}")
        if path.startswith(("data/", "All_Data/")):
            blockers.append(f"Raw dataset tracked in Git: {path}")
    for path in git(root, "diff", "--cached", "--name-only").splitlines():
        if path.startswith(("data/", "All_Data/")) or Path(path).suffix in {".npy", ".npz", ".pt", ".pth"}:
            blockers.append(f"Large/generated input staged: {path}")

    if not git(root, "config", "user.name").strip() or not git(root, "config", "user.email").strip():
        blockers.append("Git author identity not configured: use your own identity, not ubuntu@ec2")
    for relative in (
        "src/v1_mnist/component/p3_novelty_detection/README.md",
        "src/v1_mnist/tests/p3_novelty_detection/README.md",
    ):
        doc = (root / relative)
        if not doc.exists() or "168 passed" not in doc.read_text(encoding="utf-8"):
            advisories.append(f"EC2 verification summary missing or different: {relative}")

    advisories.extend([
        "PR #17 requires Fyrooz to re-review before merge; GitHub state is independent of local tests",
        "Do not claim SVHN IF/AE poor AUROC is explained: score-direction check alone is insufficient",
        "Historical classical/autoencoder numerical replay and old result-folder provenance need review",
        "FPR@95TPR definitions and prior preregistration timestamp need documented reconciliation",
        "Same fitting budget is not established for historical 10k-SVM vs 49k baseline",
        "Separate LaTeX report Issue #18 is outside this PR",
    ])
    return blockers, advisories


def main() -> None:
    """Print machine-readable assessment, retaining nonzero exit for blockers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    blockers, advisories = audit(args.root.resolve())
    if args.json:
        print(json.dumps({"blockers": blockers, "advisories": advisories}, indent=2))
    else:
        LOGGER.info("PR #17: %d blockers, %d review advisories", len(blockers), len(advisories))
        for item in blockers:
            LOGGER.error("BLOCKER: %s", item)
        for item in advisories:
            LOGGER.warning("REVIEW: %s", item)
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
