"""Save every results figure the same way: one PDF, fonts embedded as TrueType.

Project rules for figures:
- PDF only (no SVG, PNG or JPG), saved with ``bbox_inches="tight"``;
- ``pdf.fonttype = 42``, so fonts are embedded as TrueType instead of Type 3;
- very dense plots may rasterize their dense layer only (``rasterize_dense``),
  so the PDF stays small while the axes, ticks and text stay vector. The SOM
  figures do not use it: it breaks NNSOM's plots made of many small inset
  charts, and every SOM figure PDF is under 2.2 MB without it.

Figures go into a ``pdf/`` folder, the layout the project already uses
(``figures/pdf/<category>/<name>.pdf``), so existing links keep working.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Matplotlib settings used for every saved figure.
PDF_RCPARAMS = {"pdf.fonttype": 42, "ps.fonttype": 42}

# Resolution of the rasterized dense layer only (axes and text stay vector).
RASTER_DPI = 200

# Sub-folders that name a figure category (figures/pdf/<category>/<name>.pdf).
FIGURE_CATEGORIES = {
    "core",
    "class_maps",
    "feature_maps",
    "component_planes",
    "analysis_maps",
    "extended",
    "convergence",
}


def pdf_figure_path(output_path: str | Path) -> Path:
    """Where a figure asked for at ``output_path`` is saved as PDF.

    - a path inside a ``pdf/`` or (older callers) ``svg/`` folder keeps its place,
      in the ``pdf/`` folder;
    - ``<dir>/<category>/<name>`` becomes ``<dir>/pdf/<category>/<name>.pdf``;
    - anything else becomes ``<parent>/pdf/<name>.pdf``.
    Any extension given is replaced by ``.pdf``.
    """
    path = Path(output_path)
    parts = list(path.parts)
    folders = parts[:-1]
    for fmt in ("pdf", "svg"):
        if fmt in folders:
            parts[folders.index(fmt)] = "pdf"
            return Path(*parts).with_suffix(".pdf")
    if path.parent.name in FIGURE_CATEGORIES:
        return path.parent.parent / "pdf" / path.parent.name / f"{path.stem}.pdf"
    return path.parent / "pdf" / f"{path.stem}.pdf"


def rasterize_dense_layer(fig: Figure) -> None:
    """Rasterize the dense data layer: every collection (scatter points, violin bodies,
    filled areas) gets ``rasterized=True``. Axes, ticks, text, lines and single shapes
    (such as the SOM hexagon outlines) stay vector.

    Each artist is rasterized in place, so plots made of many small inset axes
    (NNSOM's violin and scatter plots) keep their layout.
    """
    for ax in fig.axes:
        for collection in ax.collections:
            collection.set_rasterized(True)


def save_pdf(
    fig: Figure,
    output_path: str | Path,
    *,
    rasterize_dense: bool = False,
    close: bool = True,
) -> Path:
    """Save ``fig`` as one PDF with embedded TrueType fonts; return the PDF path.

    Args:
        fig: the Matplotlib figure.
        output_path: where the figure is wanted; see ``pdf_figure_path``.
        rasterize_dense: rasterize the dense data layer (component planes,
            violin and box plots, scatter plots of many points).
        close: close the figure after saving.
    """
    pdf_path = pdf_figure_path(output_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if rasterize_dense:
        rasterize_dense_layer(fig)
    with plt.rc_context(PDF_RCPARAMS):
        if rasterize_dense:
            fig.savefig(pdf_path, bbox_inches="tight", dpi=RASTER_DPI)
        else:
            fig.savefig(pdf_path, bbox_inches="tight")
    if close:
        plt.close(fig)
    return pdf_path
