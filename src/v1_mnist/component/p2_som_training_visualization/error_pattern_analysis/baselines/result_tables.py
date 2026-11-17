"""Summary CSV and the LaTeX table of the baseline check, written from results.json (no hand-typed numbers)."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

METHOD_LABELS = {
    "som_cell": "SOM {h}$\\times${w} cell error rate",
    "kmeans_cell": "k-means, {cells} clusters, cell error rate",
    "knn_impurity": "kNN impurity ($k={k}$)",
    "max_softmax": "CNN max-softmax probability",
}


def _range(entry: Mapping[str, float]) -> str:
    return f"{entry['lower']:.3f}--{entry['upper']:.3f}"


def summary_rows(results: Mapping[str, Any]) -> list[dict[str, str]]:
    """One row per method with the numbers that go into the table (strings, already rounded)."""
    detection = results["error_detection"]["methods"]
    boot = results["error_detection"]["bootstrap"]
    hotspot = results["hotspot"]
    som = hotspot["som_cell"]
    km = hotspot["kmeans_cell"]
    som_conc = som["hotspot_concentration_held_out"]
    rows: list[dict[str, str]] = []
    for key in ("som_cell", "kmeans_cell", "knn_impurity", "max_softmax"):
        det = detection[key]
        if key in boot:
            auroc = f"{boot[key]['point']:.3f} ({_range(boot[key])})"
        else:
            auroc = f"{det['auroc']['mean']:.3f} ({_range(det['auroc'])})"
        row = {"method": key, "auroc": auroc, "average_precision": f"{det['average_precision']['mean']:.3f}",
               "hotspot_ratio": "--", "image_share": "--", "error_share": "--"}
        if key == "som_cell":
            row.update(hotspot_ratio=f"{som_conc['median_ratio']:.1f} ({som_conc['ratio_p2_5']:.1f}--{som_conc['ratio_p97_5']:.1f})",
                       image_share=f"{som['hotspot_shares']['mean_image_share_in_hotspots'] * 100:.1f}\\%",
                       error_share=f"{som['hotspot_shares']['mean_error_share_in_hotspots'] * 100:.1f}\\%")
        if key == "kmeans_cell":
            row.update(hotspot_ratio=f"{km['median_ratio_over_seeds']:.1f} ({km['min_ratio']:.1f}--{km['max_ratio']:.1f})",
                       image_share=f"{km['mean_image_share_in_hotspots'] * 100:.1f}\\%",
                       error_share=f"{km['mean_error_share_in_hotspots'] * 100:.1f}\\%")
        rows.append(row)
    return rows


def write_summary_csv(path: Path, results: Mapping[str, Any]) -> None:
    """Small CSV with the table rows (kept in git)."""
    rows = summary_rows(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex_table(results: Mapping[str, Any]) -> str:
    """The report's one result table: error detection and hotspot concentration side by side."""
    settings = results["settings"]
    labels = {key: text.format(h=settings["grid"][0], w=settings["grid"][1], cells=settings["num_cells"],
                               k=settings["knn_neighbors"]) for key, text in METHOD_LABELS.items()}
    som_p = results["hotspot"]["som_cell"]["hotspot_concentration_held_out"]["p_value_vs_chance"]
    km_p = results["hotspot"]["kmeans_cell"]["max_p_value_vs_chance"]
    body = [" & ".join([labels[r["method"]], r["auroc"], r["average_precision"], r["hotspot_ratio"],
                        r["image_share"], r["error_share"]]) + " \\\\ \\hline" for r in summary_rows(results)]
    caption = (
        f"Predicting which of the {results['validation_samples']:,} validation images the CNN gets wrong "
        f"({results['validation_errors']} mistakes), and how strongly each partition's hotspot cells concentrate them. "
        "AUROC range: spread over the repeated half-splits for the cell scores, bootstrap interval for kNN impurity and "
        "max-softmax (which need no fitting). Hotspot ratio: errors' share divided by images' share in the "
        f"{results['hotspot']['som_cell']['hotspot_concentration_held_out']['n_hotspots']} hotspot cells, chosen on one "
        f"half and measured on the other; k-means: median over {len(settings['kmeans_seeds'])} seeds with the lowest--highest "
        f"seed in brackets. Chance level is about 1 (p = {som_p:.3f} for the SOM, at most {km_p:.3f} for every k-means seed). "
        f"Run \\texttt{{{results['run_id'].replace('_', chr(92) + '_')}}}."
    )
    lines = [
        "\\begin{table}[H]", "\\centering", "\\renewcommand{\\arraystretch}{1.15}", "\\small",
        "\\begin{tabular}{|p{0.23\\textwidth}|p{0.19\\textwidth}|p{0.11\\textwidth}|p{0.15\\textwidth}|p{0.10\\textwidth}|p{0.10\\textwidth}|}",
        "\\hline",
        "\\textbf{Score} & \\textbf{AUROC (95\\% range)} & \\textbf{Avg. precision} & \\textbf{Hotspot ratio} & "
        "\\textbf{Images in hotspots} & \\textbf{Errors in hotspots} \\\\ \\hline",
        *body,
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        "\\label{tab:error-pattern-baselines}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def write_report_outputs(results: Mapping[str, Any], output_dir: Path) -> None:
    """Write the summary CSV and the LaTeX table next to results.json (the report is never written to)."""
    write_summary_csv(output_dir / "summary.csv", results)
    (output_dir / "error_pattern_baselines.tex").write_text(latex_table(results), encoding="utf-8")
