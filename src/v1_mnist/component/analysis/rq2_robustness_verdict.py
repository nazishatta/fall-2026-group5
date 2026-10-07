"""Apply the pre-specified RQ2 rules (docs/som/RQ2_ROBUSTNESS_CRITERIA.md) to the
*_rq2_robustness.json files: headline model first, then each robustness model.

Usage (from the code root):
  python src/v1_mnist/component/analysis/rq2_robustness_verdict.py \
      [--headline RUN_ID] [--robustness RUN_ID ...]
Writes outputs/v1_mnist/som/analysis/rq2_robustness_verdict.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_DIR = REPO_ROOT / "outputs/v1_mnist/som/analysis"
ALPHA = 0.05


def claims(stats: dict) -> dict:
    if not stats.get("testable", False):
        return {"claim_a": None, "claim_b": None}
    a = stats["claim_a_hotspots_held_out"]
    b = stats["claim_b_overlap_regions"]
    claim_a = a["p_value_vs_chance"] < ALPHA and a["ratio_p2_5"] > 1.0
    claim_b = b["p_value_purity_negative"] < ALPHA and b["p_value_entropy_positive"] < ALPHA
    return {"claim_a": bool(claim_a), "claim_b": bool(claim_b)}


def verdict(robust: dict, headline: dict) -> str:
    main = [c for c in ("claim_a", "claim_b") if headline[c]]
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
    parser = argparse.ArgumentParser(description="RQ2 robustness verdicts.")
    parser.add_argument("--headline", default="som_15x15_nb11_ep250_s42_v2")
    parser.add_argument("--robustness", nargs="+",
                        default=["som_20x20_nb15_ep250_s42_v2", "som_10x10_nb7_ep250_s42_v2"])
    args = parser.parse_args()
    load = lambda rid: json.loads((ANALYSIS_DIR / f"{rid}_rq2_robustness.json").read_text(encoding="utf-8"))
    head = claims(load(args.headline))
    show = lambda v: "not testable" if v is None else ("holds" if v else "does not hold")
    print(f"{args.headline} (headline): claim A {show(head['claim_a'])}, claim B {show(head['claim_b'])}")
    report = {"alpha": ALPHA, "headline": {"run_id": args.headline, **head}, "robustness": []}
    for rid in args.robustness:
        c = claims(load(rid))
        v = verdict(c, head)
        report["robustness"].append({"run_id": rid, **c, "verdict": v})
        print(f"{rid}: claim A {show(c['claim_a'])}, claim B {show(c['claim_b'])} -> {v}")
    (ANALYSIS_DIR / "rq2_robustness_verdict.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
