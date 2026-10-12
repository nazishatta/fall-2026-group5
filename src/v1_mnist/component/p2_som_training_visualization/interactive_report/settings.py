"""Constants shared by the interactive SOM report modules."""

TOPN = 10                  # nearest samples listed per neuron (NNSOM int_dict 'topn')
REPORT_FILENAME = "interactive_report.html"

# The page's JavaScript files in templates/js/, in the order they are joined.
# Later files use names defined by earlier ones, so the order matters.
# Keep templates/js/README.md in step with this list.
JS_PARTS = (
    "core.js",          # payload, colour ramps, small utilities, tooltip, shared state
    "views.js",         # map view definitions and hexagon geometry
    "map.js",           # map drawing, legend and the view/option/split controls
    "detail_panel.js",  # click-to-inspect panel: neuron selection, stats tiles, tab routing
    "chart_kit.js",     # small SVG chart helpers shared by the detail tabs
    "detail_tabs.js",   # one function per detail tab
    "main.js",          # page header text and first render
)
