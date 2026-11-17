# SOM convergence plots (`src/v1_mnist/component/p2_som_training_visualization/som_nnsom_training/convergence_plots/`)

Three interactive HTML charts of how a SOM trained: the quantization error (QE) per epoch, the training and validation quality, and the shrinking neighbourhood radius. Copies of the charts for the official SOM are in `demo/interactive_plots/v1_mnist/phase_2_files/convergence_plots/`.

## Files

| File | What it is |
|---|---|
| `artifacts.py` | Utilities for interactive SOM visualizations. |
| `generate_som_qe_interactive.py` ▶ | Generate an interactive SOM quantization-error convergence plot. |
| `generate_som_quality_interactive.py` ▶ | Generate an interactive train-vs-validation SOM quality summary. |
| `generate_som_radius_interactive.py` ▶ | Generate an interactive SOM neighborhood-radius schedule plot. |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
