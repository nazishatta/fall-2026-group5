"""Build the SOM re-analysis results table (tables/summary/reanalysis_results.tex).

Reads the saved CSV/JSON files of one re-analysis run; no numbers are typed in by hand.
Usage: python make_reanalysis_table.py [RESULTS_FOLDER] [OUT_FOLDER]
Default: the copied results in tables/som_reanalysis, output in tables/summary.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RUN = Path(__file__).resolve().parent / "som_reanalysis"
ARMS = ["baseline", "standard", "augmented_som", "augmented_random"]
PAIRS = ["augmented_som_vs_standard", "augmented_som_vs_augmented_random"]
# (column name, label, decimals, scale to percent points)
MEASURES = [
    ("weighted_purity", r"Weighted purity (R1, R3)", 4),
    ("low_purity_share", r"Share of neurons with purity $<0.95$", 4),
    ("class_scatter_index", r"Class scatter index", 2),
    ("topographic_error_pct", r"Topographic error, \% (R5)", 2),
    ("hotspot_error_rate", r"Hotspot error rate (R2, R4)", 4),
    ("hotspot_error_share", r"Share of errors in hotspots", 4),
    ("overall_error", r"Overall validation error", 4),
]


def signed(x: float, nd: int) -> str:
    return f"{x:+.{nd}f}".replace("-", "$-$")


def interval(row: pd.Series, nd: int) -> str:
    text = f"{signed(row.mean_difference, nd)} [{signed(row.interval_low, nd)}, {signed(row.interval_high, nd)}]"
    return f"\\textbf{{{text}}}" if bool(row.interval_excludes_zero) else text


def build(run: Path) -> list[str]:
    cells = pd.read_csv(run / "cell_measures.csv")
    pairs = pd.read_csv(run / "paired_comparisons.csv")
    cka = json.loads((run / "cka.json").read_text())["group_means"]
    arm_mean = {a: cells[cells.cell == a] if a == "baseline" else cells[cells.cell.str.startswith(a + "_seed")] for a in ARMS}
    for arm in ARMS[1:]:
        assert len(arm_mean[arm]) == 5, f"{arm}: expected 5 seeds"
    lines = [
        r"\begin{table}[H]", r"\centering", r"\renewcommand{\arraystretch}{1.12}", r"\footnotesize",
        r"{\setlength{\tabcolsep}{3pt}\begin{tabular}{|>{\raggedright\arraybackslash}p{0.18\textwidth}|C{0.08\textwidth}|C{0.085\textwidth}|C{0.09\textwidth}|C{0.09\textwidth}|C{0.185\textwidth}|C{0.185\textwidth}|}",
        r"\hline",
        r"\textbf{Measure (validation)} & \textbf{Phase 2 map} & \textbf{standard} & \textbf{augm.\ som} & \textbf{augm.\ random} & \textbf{augm.\ som $-$ standard} & \textbf{augm.\ som $-$ augm.\ random} \\ \hline",
    ]
    for col, label, nd in MEASURES:
        means = " & ".join(f"{arm_mean[a][col].mean():.{nd}f}" for a in ARMS)
        diffs = " & ".join(interval(pairs[(pairs.pair == p) & (pairs.measure == col)].iloc[0], nd) for p in PAIRS)
        lines.append(f"{label} & {means} & {diffs} \\\\ \\hline")
    lines += [
        r"\end{tabular}}",
        r"\caption{SOM re-analysis on the validation split. Columns 2 to 5 are means over the 5 CNN seeds (the Phase 2 map is a single map). The last two columns are the paired difference per seed, averaged, with its 95\% $t$-interval (4 degrees of freedom); bold means the interval excludes 0. Linear CKA between the networks' embeddings: " + f"{cka['within_standard']:.3f} within standard, {cka['within_augmented_som']:.3f} within augmented\\_som, {cka['within_augmented_random']:.3f} within augmented\\_random, {cka['between_groups']:.3f} between arms.}}",
        r"\label{tab:reanalysis}", r"\end{table}",
    ]
    return lines


def main(run: Path, out: Path) -> None:
    (out / "reanalysis_results.tex").write_text("\n".join(build(run)) + "\n", encoding="utf-8")
    logger.info("wrote %s", out / "reanalysis_results.tex")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN, Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "summary")
