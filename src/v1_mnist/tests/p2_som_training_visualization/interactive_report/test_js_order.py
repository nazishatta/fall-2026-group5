"""The report's JavaScript files are joined in the order set in settings.JS_PARTS."""

from __future__ import annotations

import re

from src.v1_mnist.component.p2_som_training_visualization.interactive_report.html_report import (
    TEMPLATE_DIR,
    load_script,
)
from src.v1_mnist.component.p2_som_training_visualization.interactive_report.settings import (
    JS_PARTS,
)

JS_DIR = TEMPLATE_DIR / "js"


def test_every_js_file_is_listed_once() -> None:
    assert len(set(JS_PARTS)) == len(JS_PARTS)
    assert set(JS_PARTS) == {p.name for p in JS_DIR.glob("*.js")}


def test_readme_lists_the_same_order() -> None:
    readme = (JS_DIR / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*(\d+)\s*\|\s*`([^`]+\.js)`", readme, flags=re.M)
    assert [int(n) for n, _ in rows] == list(range(1, len(rows) + 1))
    assert tuple(name for _, name in rows) == JS_PARTS


def test_script_joins_the_files_in_that_order() -> None:
    script = load_script()
    starts = [script.index(f"// {name} - ") for name in JS_PARTS]
    assert starts == sorted(starts)
