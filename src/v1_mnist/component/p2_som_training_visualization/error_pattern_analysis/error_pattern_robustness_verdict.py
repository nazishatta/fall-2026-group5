"""Apply the pre-specified error-pattern rules (docs/p2_som_training_visualization/error_pattern_analysis/ERROR_PATTERN_ROBUSTNESS_CRITERIA.md) to the
*_error_pattern_robustness.json files: headline model first, then each robustness model.

Usage (from the code root):
  python src/v1_mnist/component/p2_som_training_visualization/error_pattern_analysis/error_pattern_robustness_verdict.py \
      [--config CONFIG] [--headline RUN_ID] [--robustness RUN_ID ...]
By default the headline SOM is the config's selected model, the robustness SOMs are
those named by the configs in robustness_check.configs, the folder is
paths.analysis_dir and alpha is analysis.robustness.alpha.
Writes outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/robustness/error_pattern_robustness_verdict.json
(refuses if it exists: results are never overwritten).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.logging import get_logger  # noqa: E402
from src.v1_mnist.component.utils.run_settings import (  # noqa: E402
    DEFAULT_CONFIG,
    analysis_config,
    robustness_dir,
    robustness_models,
    selected_model_from_config,
)


logger = get_logger("v1_mnist.analysis.error_pattern_robustness_verdict")

# The two error-pattern findings (proposal, Objective 4: "error hotspots" and
# "overlap regions"). The criteria file calls them Claim A and Claim B.
FINDINGS = ("hotspot_concentration", "overlap_regions")


def claims(stats: dict, alpha: float) -> dict:
    if not stats.get("testable", False):
        return {name: None for name in FINDINGS}
    hot = stats["hotspot_concentration_held_out"]
    overlap = stats["overlap_regions"]
    hotspot_concentration = hot["p_value_vs_chance"] < alpha and hot["ratio_p2_5"] > 1.0
    overlap_regions = (overlap["p_value_purity_negative"] < alpha
                       and overlap["p_value_entropy_positive"] < alpha)
    return {"hotspot_concentration": bool(hotspot_concentration), "overlap_regions": bool(overlap_regions)}


def verdict(robust: dict, headline: dict) -> str:
    main = [c for c in FINDINGS if headline[c]]
    if not main:
        return "no main claim holds on the headline model; nothing to check"
    if any(robust[c] is None for c in main):
        return "not testable (too few errors)"
    held = [c for c in main if robust[c]]
    if len(held) == len(main):
        return "holds"
    if held:
        return "holds in part (fails: " + ", ".join(sorted(set(main) - set(held))) + ")"
    return "does not hold"


def main() -> None:
    parser = argparse.ArgumentParser(description="Error-pattern robustness verdicts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Headline SOM config (names the headline and robustness SOMs).")
    parser.add_argument("--headline", default=None, help="Headline run ID (default: from the config).")
    parser.add_argument("--robustness", nargs="+", default=None,
                        help="Robustness run IDs (default: from robustness_check.configs).")
    args = parser.parse_args()
    headline = args.headline or selected_model_from_config(args.config, REPO_ROOT)
    robustness = args.robustness or robustness_models(args.config, REPO_ROOT)
    alpha = float(analysis_config(args.config, REPO_ROOT).robustness.alpha)
    analysis_dir = robustness_dir(args.config, REPO_ROOT)
    out_path = analysis_dir / "error_pattern_robustness_verdict.json"
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists. Results are never overwritten; "
            "move or delete that file to redo the verdict."
        )
    load = lambda rid: json.loads((analysis_dir / f"{rid}_error_pattern_robustness.json").read_text(encoding="utf-8"))
    head = claims(load(headline), alpha)
    show = lambda v: "not testable" if v is None else ("holds" if v else "does not hold")
    logger.info("%s (headline): hotspot concentration %s, overlap regions %s",
                headline, show(head["hotspot_concentration"]), show(head["overlap_regions"]))
    report = {"alpha": alpha, "headline": {"run_id": headline, **head}, "robustness": []}
    for rid in robustness:
        c = claims(load(rid), alpha)
        v = verdict(c, head)
        report["robustness"].append({"run_id": rid, **c, "verdict": v})
        logger.info("%s: hotspot concentration %s, overlap regions %s -> %s",
                    rid, show(c["hotspot_concentration"]), show(c["overlap_regions"]), v)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
