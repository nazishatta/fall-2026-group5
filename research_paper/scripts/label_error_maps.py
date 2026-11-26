"""Add a key and hotspot labels to the existing validation error maps (no new experiments).

The input figures are the project's own NNSOM "complex hit" maps
(outputs/<study>/p2_som_training_visualization/error_pattern_analysis/figures/<run>/pdf/analysis_maps/
validation_complex_error_map.pdf). They are kept unchanged and placed on a larger page; this script adds
(1) a bold outline and a "true class -> predicted class" label on every hotspot cell, and (2) a key below.
Cell positions are read from the vector paths of the original PDF and checked against the saved results:
every class must map to exactly one fill colour.

Usage (from the repository root):
    python research_paper/scripts/label_error_maps.py --out research_paper/Latex/Fig
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pymupdf

logger = logging.getLogger("label_error_maps")

MNIST_CLASSES = [str(i) for i in range(10)]
FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
FASHION_SHORT = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
KEY_HEIGHT = 120.0
GRID = 15


@dataclass(frozen=True)
class Study:
    """Saved results of one dataset."""

    key: str
    root: str
    run: str
    classes: list[str]
    short: list[str]


STUDIES = [
    Study("mnist", "v1_mnist", "som_15x15_nb11_ep250_s42_v2", MNIST_CLASSES, MNIST_CLASSES),
    Study("fashion_mnist", "v1_fashion_mnist", "som_15x15_nb11_ep250_s42", FASHION_CLASSES, FASHION_SHORT),
]


def cell_polygons(page: pymupdf.Page) -> dict[int, list[tuple[float, float]]]:
    """Return the outline polygon of every cell, keyed by neuron id (row 0 = bottom row)."""

    grey = [d for d in page.get_drawings()
            if len(d["items"]) == 6 and d["rect"].width < 40 and d.get("color") is not None
            and abs(d["color"][0] - 0.8) < 0.02]
    centres = [((d["rect"].x0 + d["rect"].x1) / 2, (d["rect"].y0 + d["rect"].y1) / 2, d) for d in grey]
    if len(centres) != GRID * GRID:
        raise ValueError(f"expected {GRID * GRID} cell outlines, found {len(centres)}")
    centres.sort(key=lambda c: -c[1])
    rows: list[list[tuple[float, float, dict]]] = [[centres[0]]]
    for c in centres[1:]:
        if abs(c[1] - rows[-1][-1][1]) < 3:
            rows[-1].append(c)
        else:
            rows.append([c])
    polys: dict[int, list[tuple[float, float]]] = {}
    for r_idx, row in enumerate(rows):
        for c_idx, (_, _, d) in enumerate(sorted(row, key=lambda c: c[0])):
            pts = [it[1] for it in d["items"] if it[0] == "l"]
            polys[r_idx * GRID + c_idx] = [(p.x, p.y) for p in pts]
    return polys


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def class_colours(page: pymupdf.Page, polys: dict[int, list[tuple[float, float]]],
                  dominant: dict[int, int]) -> dict[int, tuple[float, float, float]]:
    """Read the fill colour of each class from the PDF and check it is unique per class."""

    fills = [d for d in page.get_drawings()
             if len(d["items"]) == 6 and d["rect"].width < 40 and d.get("fill") is not None
             and tuple(round(v, 2) for v in d["fill"]) != (1.0, 1.0, 1.0)]
    seen: dict[int, Counter[tuple[float, float, float]]] = defaultdict(Counter)
    for neuron, cls in dominant.items():
        cx = float(np.mean([p[0] for p in polys[neuron]]))
        cy = float(np.mean([p[1] for p in polys[neuron]]))
        near = [d for d in fills if abs((d["rect"].x0 + d["rect"].x1) / 2 - cx) < 1.5
                and abs((d["rect"].y0 + d["rect"].y1) / 2 - cy) < 1.5]
        if near:
            best = max(near, key=lambda d: d["rect"].width)
            seen[cls][tuple(round(v, 3) for v in best["fill"])] += 1
    colours: dict[int, tuple[float, float, float]] = {}
    for cls, counter in seen.items():
        if len(counter) != 1:
            raise ValueError(f"class {cls} has several fill colours: {counter}")
        colours[cls] = next(iter(counter))
    return colours


def label_one(study: Study, repo: Path, out: Path) -> Path:
    base = repo / "outputs" / study.root / "p2_som_training_visualization"
    src_pdf = (base / "error_pattern_analysis/figures" / study.run / "pdf/analysis_maps/validation_complex_error_map.pdf")
    geo = read_rows(base / "error_pattern_analysis" / study.run / "tables/val_error_geography.csv")
    mis = read_rows(base / "error_pattern_analysis" / study.run / "tables/val_misclassified_samples.csv")
    inter = np.loadtxt(base / "cluster_analysis" / study.run / "tables/val_class_neuron_intersection.csv", delimiter=",")

    hot = {int(r["neuron_id"]) for r in geo if r["hotspot"] == "True"}
    pairs: dict[int, Counter[tuple[int, int]]] = defaultdict(Counter)
    for row in mis:
        if row["hotspot"] == "True":
            pairs[int(row["bmu"])][(int(row["true_class"]), int(row["predicted_class"]))] += 1
    dominant = {int(r["neuron_id"]): int(np.argmax(inter[:, int(r["neuron_id"])]))
                for r in geo if float(r["support"]) > 0}

    src = pymupdf.open(src_pdf)
    page = src[0]
    polys = cell_polygons(page)
    colours = class_colours(page, polys, dominant)
    width, height = page.rect.width, page.rect.height

    doc = pymupdf.open()
    new = doc.new_page(width=width, height=height + KEY_HEIGHT)
    new.show_pdf_page(pymupdf.Rect(0, 0, width, height), src, 0)
    font_file = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
    bold_file = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf")
    new.insert_font(fontname="dv", fontfile=font_file)
    new.insert_font(fontname="dvb", fontfile=bold_file)
    font = pymupdf.Font(fontfile=font_file)

    boxes: list[pymupdf.Rect] = []
    cell_rects = {n: pymupdf.Rect(min(p[0] for p in polys[n]), min(p[1] for p in polys[n]),
                                  max(p[0] for p in polys[n]), max(p[1] for p in polys[n])) for n in hot}
    for n in sorted(hot):
        new.draw_polyline(polys[n] + [polys[n][0]], color=(0, 0, 0), width=2.4, closePath=True)
    for n in sorted(hot, key=lambda k: cell_rects[k].y0):
        (t, p), _ = pairs[n].most_common(1)[0]
        text = f"{study.short[t]}→{study.short[p]}"
        size = 8.5
        tw = font.text_length(text, fontsize=size)
        rect = cell_rects[n]
        cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
        options = [(cx - tw / 2, rect.y0 - size - 3), (cx - tw / 2, rect.y1 + 2),
                   (rect.x1 + 2, cy - size / 2), (rect.x0 - tw - 2, cy - size / 2),
                   (cx - tw / 2, rect.y0 - 2 * size - 5), (cx - tw / 2, rect.y1 + size + 4)]
        chosen = options[0]
        for x, y in options:
            cand = pymupdf.Rect(x - 1.5, y - 1, x + tw + 1.5, y + size + 1.5)
            clash = any(cand.intersects(b) for b in boxes) or any(
                cand.intersects(cell_rects[m]) for m in hot if m != n)
            if not clash and cand.x0 > 0 and cand.x1 < width and cand.y0 > 24:
                chosen = (x, y)
                break
        x, y = chosen
        box = pymupdf.Rect(x - 1.5, y - 1, x + tw + 1.5, y + size + 1.5)
        boxes.append(box)
        new.draw_rect(box, color=(0.2, 0.2, 0.2), fill=(1, 1, 1), width=0.4, fill_opacity=0.92)
        new.insert_text((x, y + size - 1.2), text, fontname="dv", fontsize=size, color=(0, 0, 0))

    # crop to the map (drops the original title, whose content is in the key and caption), then draw the key below
    all_polys = [pt for poly in polys.values() for pt in poly]
    gx0, gx1 = min(p[0] for p in all_polys), max(p[0] for p in all_polys)
    gy0, gy1 = min(p[1] for p in all_polys), max(p[1] for p in all_polys)
    x0 = min([gx0] + [b.x0 for b in boxes]) - 6
    x1 = max([gx1] + [b.x1 for b in boxes]) + 6
    top = min([gy0] + [b.y0 for b in boxes]) - 4
    kw = x1 - x0
    y0 = gy1 + 8
    new.insert_text((x0 + 4, y0 + 9), "Key", fontname="dvb", fontsize=10, color=(0, 0, 0))
    col_w = (kw - 8) / 5
    for i in range(len(study.classes)):
        row, col = divmod(i, 5)
        x = x0 + 4 + col * col_w
        y = y0 + 16 + row * 15
        new.draw_rect(pymupdf.Rect(x, y, x + 11, y + 10), color=(0.3, 0.3, 0.3), fill=colours[i], width=0.4)
        new.insert_text((x + 14, y + 9), study.classes[i], fontname="dv", fontsize=9, color=(0, 0, 0))
    notes = ("Fill color: most common true class of the cell. Edge color: most common wrong prediction "
             "(same color code). Edge width: how strongly errors concentrate in the cell. "
             "Number: validation images in the cell. Bold black outline: hotspot cell, labeled with its "
             "most frequent confusion (true class \u2192 predicted class).")
    note_rect = pymupdf.Rect(x0 + 4, y0 + 48, x1 - 4, y0 + 110)
    new.insert_textbox(note_rect, notes, fontname="dv", fontsize=8.5, color=(0, 0, 0))
    page_h = new.rect.height  # set_mediabox takes PDF coordinates (origin at the bottom left)
    new.set_mediabox(pymupdf.Rect(x0, page_h - (y0 + 94), x1, page_h - top))
    target = out / f"error_map_{study.key}_labeled.pdf"
    doc.save(target, garbage=4, deflate=True)
    logger.info("wrote %s (%d hotspot labels)", target, len(hot))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    for study in STUDIES:
        label_one(study, args.repo, args.out)


if __name__ == "__main__":
    main()
