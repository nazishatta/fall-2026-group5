# Cluster analysis results (`outputs/v1_mnist/p2_som_training_visualization/cluster_analysis/`)

Results of Phase 2, Week 4 (code: `src/v1_mnist/component/p2_som_training_visualization/cluster_analysis/`), one folder per SOM.

| Path | What is inside |
|---|---|
| `som_15x15_nb11_ep250_s42_v2/cluster_summary.json` | Purity, entropy and the label-shuffle control for the headline map |
| `som_15x15_nb11_ep250_s42_v2/tables/` | Per-neuron cluster metrics and class-neuron intersection (train and validation), validation confusion counts on the map |
| `som_15x15_nb11_ep250_s42_v2/assignments/` (+ `.dvc`) | The best-matching neuron of every train and validation image (DVC) |
| `som_15x15_nb11_ep250_s42_v2/metadata/` | SHA-256 fingerprints of the inputs and of the files written |
| `figures/` (+ `figures.dvc`) | Class maps, purity map, class pie and stem charts (PDF, DVC) |
| `logs/` | Logs of the figure scripts |

Main result: validation purity 0.9819, against 0.1738 when the labels are shuffled (1,000 shuffles).
