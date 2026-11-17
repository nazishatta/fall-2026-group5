# Interactive SOM report (`src/v1_mnist/component/p2_som_training_visualization/interactive_report/`)

Builds one self-contained HTML page that shows the official SOM: the map, which digits each neuron holds, where the mistakes are, and sample images. It covers Phase 2 (SOM training, cluster analysis and error pattern analysis) in one place.

Runner: `pipeline/p2_som_training_visualization/run_interactive_report.py`. Result: `outputs/v1_mnist/p2_som_training_visualization/interactive_report/interactive_report.html` (a copy is in `demo/interactive_plots/`).

## Folders

| Folder | What is inside |
|---|---|
| [`templates/`](templates/README.md) | The HTML page, styles and JavaScript that `html_report.py` fills with the SOM data to make the interactive report. |

## Files

| File | What it is |
|---|---|
| `html_report.py` | Fill the HTML template with the payload and write the report file(s). |
| `model_io.py` | Load a saved NNSOM model (same logic as utils/som_figure_setup.py, load_model_and_grid). |
| `neuron_stats.py` | Per-neuron statistics for the interactive report. |
| `numeric_utils.py` | Small NumPy helpers: CuPy-safe conversion and JSON-friendly rounding. |
| `report_payload.py` | Assemble the JSON payload that the HTML page renders. |
| `settings.py` | Constants shared by the interactive SOM report modules. |
| `visualize_som_interactive.py` ▶ | Build the interactive SOM report (single self-contained HTML page). |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
