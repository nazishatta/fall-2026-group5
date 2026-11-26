"""Prepare the files for the Neurocomputing submission.

Three steps, all on copies (the result figures in outputs/ are never changed):

1. ``strip-titles``: remove the title line printed at the top of some result figures in
   ``Latex/Fig/``. The guide for authors asks that a figure's title appear in its caption, not
   on the figure itself. Only the page box is moved, so nothing inside the plot changes, and a
   figure whose title is already gone is left alone.
2. ``highlights``: write ``Latex/Highlights.docx`` from ``Latex/Highlights.txt``.
3. ``package``: copy every figure, in the order it appears in the paper, to
   ``submission_package/Figure_<n>.pdf`` (``Figure_<n>a``/``b`` for two-panel figures,
   ``Figure_A<n>`` in the appendix), with a copy of the .tex that uses these names, the .bbl,
   the .bib, the highlights and the cover letter. The folder is flat, as the submission
   system expects.

Run from the repository root:
    python research_paper/scripts/package_submission.py all
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import tempfile
from pathlib import Path

import pymupdf
from docx import Document

logger = logging.getLogger(__name__)

PAPER_DIR = Path("research_paper/Latex")
FIG_DIR = PAPER_DIR / "Fig"
PACKAGE_DIR = Path("research_paper/submission_package")
TEX_NAME = "neurocomputing_paper.tex"
SOURCE_FILES = [TEX_NAME, "neurocomputing_paper.bbl", "references.bib", "Highlights.docx", "cover_letter.tex"]

# figure file -> first word of the title printed at its top (only that line is removed)
TITLED_FIGURES: dict[str, str] = {
    "mnist_p2_u_matrix.pdf": "SOM",
    "fashion_mnist_p2_u_matrix.pdf": "SOM",
    "svhn_som_overlay_mnist.pdf": "SVHN",
    "svhn_som_overlay_fashion_mnist.pdf": "SVHN",
    "hotspot_examples_mnist.pdf": "Representative",
    "hotspot_examples_fashion_mnist.pdf": "Representative",
    "mnist_p4_map_measures_by_arm.pdf": "Validation",
    "fashion_mnist_p4_map_measures_by_arm.pdf": "Validation",
}


def strip_title(path: Path, first_word: str) -> bool:
    """Move the top of the page box below the title line. Returns True if the file changed."""
    doc = pymupdf.open(path)
    page = doc[0]
    words = sorted(page.get_text("words"), key=lambda w: (w[1], w[0]))
    if not words or words[0][4] != first_word:
        logger.info("%s: no title found, left unchanged", path.name)
        return False
    title_bottom = max(w[3] for w in words if abs(w[1] - words[0][1]) < 1.0)
    below = [w[1] for w in words if w[1] > title_bottom]
    cut = title_bottom + (min(2.0, (min(below) - title_bottom) / 2) if below else 2.0)
    height = page.mediabox.height  # set_mediabox uses PDF coordinates (origin at the bottom left)
    box = page.mediabox
    page.set_mediabox(pymupdf.Rect(box.x0, box.y0, box.x1, height - cut))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / path.name
        doc.save(out, garbage=4, deflate=True)
        doc.close()
        shutil.copyfile(out, path)  # overwrite in place (no delete needed)
    logger.info("%s: removed title (%.1f pt from the top)", path.name, cut)
    return True


def write_highlights_docx(txt: Path, docx_path: Path) -> None:
    lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for ln in lines:
        if len(ln) > 85:
            raise ValueError(f"highlight longer than 85 characters ({len(ln)}): {ln}")
    if not 3 <= len(lines) <= 5:
        raise ValueError(f"expected 3 to 5 highlights, found {len(lines)}")
    document = Document()
    document.add_heading("Highlights", level=1)
    for ln in lines:
        document.add_paragraph(ln, style="List Bullet")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / docx_path.name
        document.save(out)
        shutil.copyfile(out, docx_path)
    logger.info("wrote %s (%d highlights)", docx_path, len(lines))


def figure_files_in_order(tex: Path) -> list[tuple[str, list[str]]]:
    """Return (figure number, [files]) for every figure environment, in document order."""
    text = tex.read_text(encoding="utf-8")
    appendix_at = text.find("\\appendix")
    result: list[tuple[str, list[str]]] = []
    main_n = app_n = 0
    for match in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", text, re.S):
        files: list[str] = []
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", match.group(0)):
            if name not in files:
                files.append(name)
        if appendix_at != -1 and match.start() > appendix_at:
            app_n += 1
            number = f"A{app_n}"
        else:
            main_n += 1
            number = str(main_n)
        result.append((number, files))
    return result


def build_package(paper_dir: Path, package_dir: Path) -> None:
    """Write a flat folder that can be uploaded as it is.

    The submission system puts every uploaded file in one folder and builds the PDF from the
    .tex, so the figure names used by ``\\includegraphics`` must match the uploaded file names.
    The package therefore holds a copy of the .tex in which each figure file is renamed to
    ``Figure_<n>.pdf`` and ``\\graphicspath`` points to the same folder.
    """
    package_dir.mkdir(parents=True, exist_ok=True)
    tex_text = (paper_dir / TEX_NAME).read_text(encoding="utf-8")
    rows = ["Figure file\tSource file in Latex/Fig"]
    renames: dict[str, str] = {}
    for number, files in figure_files_in_order(paper_dir / TEX_NAME):
        for i, name in enumerate(files):
            suffix = "abcdefgh"[i] if len(files) > 1 else ""
            target = f"Figure_{number}{suffix}.pdf"
            shutil.copyfile(paper_dir / "Fig" / name, package_dir / target)
            renames[name] = target
            rows.append(f"{target}\t{name}")
            logger.info("%s <- %s", target, name)
    for name, target in renames.items():
        tex_text = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?)\{" + re.escape(name) + r"\}",
                          lambda m, t=target: f"{m.group(1)}{{{t}}}", tex_text)
    tex_text = tex_text.replace("\\graphicspath{{Fig/}}", "\\graphicspath{{./}}")
    (package_dir / TEX_NAME).write_text(tex_text, encoding="utf-8")
    (package_dir / "figure_list.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for name in SOURCE_FILES:
        if name == TEX_NAME:
            continue
        src = paper_dir / name
        if src.exists():
            shutil.copyfile(src, package_dir / name)
        else:
            logger.warning("%s not found; build the paper first so the .bbl exists", src)
    logger.info("package written to %s", package_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["strip-titles", "highlights", "package", "all"])
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    parser.add_argument("--package-dir", type=Path, default=PACKAGE_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.step in ("strip-titles", "all"):
        for name, word in TITLED_FIGURES.items():
            strip_title(args.paper_dir / "Fig" / name, word)
    if args.step in ("highlights", "all"):
        write_highlights_docx(args.paper_dir / "Highlights.txt", args.paper_dir / "Highlights.docx")
    if args.step in ("package", "all"):
        build_package(args.paper_dir, args.package_dir)


if __name__ == "__main__":
    main()
