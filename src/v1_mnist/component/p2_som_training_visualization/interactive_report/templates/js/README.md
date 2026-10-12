# Interactive report: JavaScript files

These files hold the browser-side code of the interactive SOM report. They are
not loaded separately. `html_report.py` joins them into one function scope, in
the order below, and puts the result inside the single, self-contained HTML
page. Because they share one scope, a file can use anything defined in a file
listed before it. That is why the order matters.

| Order | File | What it does |
|---|---|---|
| 1 | `core.js` | Reads the data embedded in the page; colour ramps, small utilities, tooltip, shared state |
| 2 | `views.js` | Map view definitions and hexagon geometry |
| 3 | `map.js` | Draws the map; legend and the view, option and split controls |
| 4 | `detail_panel.js` | Click-to-inspect panel: selecting a neuron, stats tiles, switching tabs |
| 5 | `chart_kit.js` | Small SVG chart helpers (axes, ticks, bars) used by the detail tabs |
| 6 | `detail_tabs.js` | One function per detail tab (classes, histogram, box, violin, scatter, weights, errors, samples) |
| 7 | `main.js` | Page header text and the first render |

The order itself is set in code, in `JS_PARTS` in `../../settings.py`. This
README must list the same files in the same order; a test checks this.

**Adding a file:** add it to `JS_PARTS` and to this table, at the point after
every file it depends on. The report refuses to build if a `.js` file here is
missing from `JS_PARTS`, or the other way round.

Do not use `import`/`export` in these files.
