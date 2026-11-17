# Tests: cluster analysis (`src/v1_mnist/tests/p2_som_training_visualization/cluster_analysis/`)

Tests for the code in `component/p2_som_training_visualization/cluster_analysis/`. Run them with `python -m pytest src/v1_mnist/tests/p2_som_training_visualization/cluster_analysis -q` from the code root.

## Files

| File | What it is |
|---|---|
| `test_bmu_assignment_and_entropy.py` | Cluster Analysis: BMU assignment and per-neuron class-mixing helpers. |
| `test_cluster_metrics.py` | Neuron entropy, cluster purity and the class-neuron intersection, including bad inputs. |
| `test_confusion_overlay.py` | Confusion overlay on the SOM map: counts per neuron, ordering, figure and no-overwrite rule. |
| `test_heatmap_figure.py` | Cluster Analysis: SOM heatmaps are saved as one PDF (no SVG). |
| `test_purity_label_shuffle.py` | Cluster analysis: label-shuffle comparison for the sample-weighted cluster purity. |
| `test_qe_model_check.py` | Cluster Analysis: the quantization-error model check. |
