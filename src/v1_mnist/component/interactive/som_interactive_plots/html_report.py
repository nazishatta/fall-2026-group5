"""Fill the HTML template with the payload and write the report file(s).

The browser-side code lives in ``templates/``:

    report_page.html   page skeleton with __STYLE__, __SCRIPT__,
                       __TITLE__ and __SOM_PAYLOAD__ placeholders
    report_style.css   all CSS
    js/NN_*.js         the page logic, split by responsibility

The js files are fragments: they are joined in numeric order inside one
function scope so they can share variables, then inlined into the page.
The result is ONE self-contained .html file that works offline.
"""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_script() -> str:
    """Join templates/js/*.js (numeric order) into one IIFE."""

    parts = [_read(p) for p in sorted((TEMPLATE_DIR / "js").glob("*.js"))]
    if not parts:
        raise FileNotFoundError(f"No .js templates in {TEMPLATE_DIR / 'js'}")
    return "(() => {\n\"use strict\";\n" + "\n".join(parts) + "\n})();"


def render_html(payload: dict) -> str:
    """Return the complete report page as a string."""

    data_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    data_json = data_json.replace("</", "<\\/")  # never close the <script> early

    title = f"SOM explorer \u2014 {payload['meta']['model']}"

    page = _read(TEMPLATE_DIR / "report_page.html")
    page = page.replace("__STYLE__", _read(TEMPLATE_DIR / "report_style.css"))
    page = page.replace("__SCRIPT__", load_script())
    page = page.replace("__TITLE__", title)
    # payload last so nothing inside the data can be mistaken for a placeholder
    return page.replace("__SOM_PAYLOAD__", data_json)


def write_report(payload: dict, destinations: list[Path]) -> list[Path]:
    """Render once and write the same file to every destination path."""

    html = render_html(payload)
    written: list[Path] = []
    for dest in destinations:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(html, encoding="utf-8")
        written.append(dest)
    return written
