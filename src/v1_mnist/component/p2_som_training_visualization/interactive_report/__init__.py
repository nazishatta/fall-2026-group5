"""Interactive (self-contained HTML) report for a trained SOM.

Package layout - one file per responsibility:

    settings.py        constants shared by the modules
    model_io.py        load the pickled SOM + read its grid size
    numeric_utils.py   NumPy/CuPy helpers and JSON-safe rounding
    neuron_stats.py    per-neuron numbers (hits, classes, errors, edges)
    report_payload.py  assemble everything the page needs into one dict
    html_report.py     fill the HTML template and write the file(s)
    templates/         report_page.html, report_style.css and js/*.js
                       (the browser-side code; js/README.md lists the
                       order the js files are joined in)

Run it with ``src/v1_mnist/component/p2_som_training_visualization/interactive_report/visualize_som_interactive.py``.
"""

from .html_report import render_html, write_report
from .model_io import load_model_and_grid
from .neuron_stats import assign_bmus, neighbour_edges, split_summary
from .report_payload import build_payload

__all__ = [
    "assign_bmus",
    "build_payload",
    "load_model_and_grid",
    "neighbour_edges",
    "render_html",
    "split_summary",
    "write_report",
]
