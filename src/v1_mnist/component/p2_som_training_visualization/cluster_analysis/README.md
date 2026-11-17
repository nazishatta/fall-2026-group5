# Cluster analysis (`src/v1_mnist/component/p2_som_training_visualization/cluster_analysis/`)

Proposal step: Cluster Analysis. Finds the best-matching neuron of every image, then measures how pure each neuron is (one digit or many), its entropy, and the class-neuron intersection. A label-shuffle control (1,000 shuffles) shows the purity is far above chance. Draws the class maps, purity map and confusion overlay.

Runner: `pipeline/p2_som_training_visualization/run_cluster_analysis.py`. Results: `outputs/v1_mnist/p2_som_training_visualization/cluster_analysis/`.

## Files

| File | What it is |
|---|---|
| `cluster_figures.py` ▶ | Figures for the selected SOM (Cluster Analysis). |
| `cluster_heatmaps.py` ▶ | Heatmaps of the SOM (Cluster Analysis). |
| `cluster_metrics.py` | SOM cluster-structure metrics (Cluster Analysis). |
| `cluster_outputs.py` | File outputs of the cluster analysis: the per-neuron table and the run metadata. |
| `confusion_overlay.py` ▶ | Where the CNN's most frequent confusions land on the SOM map. |
| `run_cluster_analysis.py` ▶ | Cluster Analysis. |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
