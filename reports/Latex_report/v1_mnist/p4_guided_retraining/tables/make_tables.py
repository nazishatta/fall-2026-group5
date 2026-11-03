"""Build the two LaTeX tables of the Phase 4 report from the saved result tables."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("make_tables")

# Macro-F1 guard values (guided arm minus control) as printed in test_verdicts.md.
GUARD = {"C1": ("-0.0044", False), "C2": ("-0.0040", False), "A1": ("+0.0003", True), "A2": ("+0.0010", True)}


def pp(x: float, digits: int = 1, sign: bool = False) -> str:
    """Format a proportion as percentage points."""
    s = f"{100 * x:+.{digits}f}" if sign else f"{100 * x:.{digits}f}"
    return s.replace("-", "$-$")


def ci(row: pd.Series, digits: int = 2) -> str:
    return f"[{pp(row.ci_low, digits, True)}, {pp(row.ci_high, digits, True)}]"


def get(df: pd.DataFrame, comparison: str, subset: str) -> pd.Series:
    sel = df[(df.comparison == comparison) & (df.subset == subset)]
    assert len(sel) == 1, (comparison, subset)
    return sel.iloc[0]


def write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("wrote %s", path)


def arm_table(cur: pd.DataFrame, aug: pd.DataFrame, out: Path) -> None:
    def row(r: pd.Series, label: str) -> str:
        return (
            f"{label} & {100 * (1 - r.overall_error):.2f} ({100 * r.overall_std:.2f}) & "
            f"{pp(r.hotspot_error)} ({pp(r.hotspot_std)}) & {pp(r.non_hotspot_error, 2)} ({pp(r.non_hotspot_std, 2)}) \\\\"
        )

    lines = [
        r"\begin{table}[H]", r"\centering", r"\renewcommand{\arraystretch}{1.1}", r"\small",
        r"\begin{tabular}{|l|C{0.20\textwidth}|C{0.20\textwidth}|C{0.20\textwidth}|}", r"\hline",
        r"\textbf{Arm} & \textbf{Test accuracy (\%)} & \textbf{Hotspot error (\%)} & \textbf{Non-hotspot error (\%)} \\ \hline",
        row(cur[cur.arm == "standard"].iloc[0], r"\textbf{standard} (baseline, CF1)") + r" \hline",
        r"\multicolumn{4}{|l|}{\textit{Curriculum}} \\ \hline",
    ]
    for arm in ["curriculum_som", "curriculum_random", "curriculum_msp"]:
        lines.append(row(cur[cur.arm == arm].iloc[0], arm.replace("_", r"\_")) + r" \hline")
    lines.append(r"\multicolumn{4}{|l|}{\textit{Targeted augmentation}} \\ \hline")
    for arm in ["augmented_som", "augmented_random", "augmented_msp", "augmented_hotspot",
                "augmented_random_hotspot", "augmented_som_paced", "augmented_random_paced"]:
        lines.append(row(aug[aug.arm == arm].iloc[0], arm.replace("_", r"\_")) + r" \hline")
    lines += [
        r"\end{tabular}",
        r"\caption{Test results, mean (standard deviation) over 5 seeds. Hotspot error is measured on the 146 test images that fall on hotspot neurons, non-hotspot error on the other 10,354. Both experiments use the same \texttt{standard} arm.}",
        r"\label{tab:arms}", r"\end{table}",
    ]
    write(out / "arm_results.tex", lines)


def criteria_table(cur: pd.DataFrame, aug: pd.DataFrame, out: Path) -> None:
    def line(cid: str, text: str, r: pd.Series, verdict: str, guard: bool = True) -> str:
        g = ""
        if cid in GUARD:
            g = f"{GUARD[cid][0].replace('-', '$-$')} ({'holds' if GUARD[cid][1] else 'fails'})"
        return f"{cid} & {text} & {pp(r.mean_difference, 2, True)} {ci(r)} & {g} & \\textbf{{{verdict}}} \\\\ \\hline"

    c1 = get(cur, "curriculum_som minus standard", "overall")
    c2 = get(cur, "curriculum_som minus curriculum_random", "hotspot")
    a1 = get(aug, "augmented_som minus augmented_random", "hotspot")
    a2 = get(aug, "augmented_som minus standard", "hotspot")
    a3 = get(aug, "augmented_som minus standard", "non_hotspot")
    r1 = get(aug, "augmented_som minus augmented_msp", "hotspot")
    r2 = get(aug, "augmented_som_paced minus augmented_som", "hotspot")
    r3 = get(aug, "augmented_hotspot minus augmented_random_hotspot", "hotspot")
    assert a3.ci_high <= 0.002 and a2.excludes_zero and a1.excludes_zero
    assert not (c1.mean_difference < 0) and not (c2.mean_difference < 0)

    lines = [
        r"\begin{table}[H]", r"\centering", r"\renewcommand{\arraystretch}{1.12}", r"\small",
        r"{\setlength{\tabcolsep}{3pt}\begin{tabular}{|C{0.06\textwidth}|>{\raggedright\arraybackslash}p{0.38\textwidth}|C{0.22\textwidth}|C{0.14\textwidth}|C{0.10\textwidth}|}", r"\hline",
        r"\textbf{ID} & \textbf{Comparison (error on)} & \textbf{Difference in points [95\% CI]} & \textbf{Macro-F1 guard} & \textbf{Verdict} \\ \hline",
        line("C1", r"curriculum\_som $-$ standard (all test images)", c1, "not met"),
        line("C2", r"curriculum\_som $-$ curriculum\_random (hotspots)", c2, "not met"),
        line("A1", r"augmented\_som $-$ augmented\_random (hotspots)", a1, "met"),
        line("A2", r"augmented\_som $-$ standard (hotspots)", a2, "met"),
        line("A3", r"augmented\_som $-$ standard (non-hotspot images; limit: upper end $\le 0.2$)", a3, "met"),
        r"\multicolumn{5}{|l|}{\textit{Also reported (not success criteria)}} \\ \hline",
        line("R1", r"augmented\_som $-$ augmented\_msp (hotspots)", r1, "tie"),
        line("R2", r"augmented\_som\_paced $-$ augmented\_som (hotspots)", r2, "worse"),
        line("R3", r"augmented\_hotspot $-$ augmented\_random\_hotspot (hotspots)", r3, "tie"),
        r"\end{tabular}}",
        r"\caption{The success criteria, fixed before the test split was used. Negative differences mean fewer errors for the first arm; 1 test hotspot image is about 0.7 points. Intervals: paired bootstrap over test images, mean of 5 seeds. The macro-F1 guard is the first arm's macro-F1 minus the second arm's; it may not fall below $-0.0016$.}",
        r"\label{tab:criteria}", r"\end{table}",
    ]
    write(out / "criteria_results.tex", lines)


def main(root: Path) -> None:
    cur_s = pd.read_csv(root / "curriculum_test_summary.csv")
    aug_s = pd.read_csv(root / "augmentation_test_summary.csv")
    cur_c = pd.read_csv(root / "curriculum_test_comparisons.csv")
    aug_c = pd.read_csv(root / "augmentation_test_comparisons.csv")
    arm_table(cur_s, aug_s, root)
    criteria_table(cur_c, aug_c, root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
