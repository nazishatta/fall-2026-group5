# `cluster_analysis` results (Fashion-MNIST)

Phase 2, cluster analysis of the headline 15x15 map (code: `src/v1_mnist/component/p2_som_training_visualization/cluster_analysis/`, shared with MNIST).

| Path | What is inside |
|---|---|
| `som_15x15_nb11_ep250_s42/cluster_summary.json` | Purity, entropy and the label-shuffle control |
| `som_15x15_nb11_ep250_s42/tables/` | Per-neuron cluster metrics and class-neuron intersection (train and validation); validation confusion counts on the map |
| `som_15x15_nb11_ep250_s42/figures/` (+ `figures.dvc`) | Most common class per neuron (train and validation), hit counts, purity, entropy and the confusion overlay (PDF, DVC) |
| `som_15x15_nb11_ep250_s42/assignments/` (+ `assignments.dvc`) | The best-matching neuron of every train and validation image (DVC) |
| `som_15x15_nb11_ep250_s42/metadata/` | SHA-256 fingerprints of the inputs and of the files written |
| `figures/` (+ `figures.dvc`) | Class maps, purity map, class pie and stem charts (PDF, DVC) |

Main result: validation purity 0.893, against 0.173 when the labels are shuffled (1,000 shuffles, largest 0.180).
