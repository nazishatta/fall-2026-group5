# `feature_layer_ablation` results (Fashion-MNIST)

SOMs on two layers of the frozen Phase 1 LeNet-5, `fc1` (120 numbers) and `fc2` (84 numbers), with the headline recipe and SOM seeds 1 to 5. Code: `src/v1_mnist/component/p2_som_training_visualization/som_nnsom_training/layer_ablation/`. Protocol: `src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/FEATURE_LAYER_ABLATION_PROTOCOL.md`, followed unchanged.

Run folder `feature_layer_ablation_som15x15_nb11_ep250_seeds5/`:

| Path | What is inside |
|---|---|
| `results.json` | The three statements (fc1 minus fc2 with 95% intervals and verdicts), the means per layer and the fc2 reproduction check |
| `tables/statements.csv` | One row per statement |
| `tables/cell_measures.csv` | One row per map (layer and SOM seed) |
| `metrics/<layer>_seed<k>.json` | All measures of one map |
| `metadata/inputs.json` | The inputs and their SHA-256 fingerprints |
| `embeddings/` (+ `embeddings.dvc`), `logs/` (+ `logs.dvc`) | The fc1 and fc2 embeddings of train, validation and SVHN, and the logs (DVC) |

Main result: `fc2` locates the network's mistakes better (error-cell AUROC difference -0.055 [-0.066, -0.043]) and is purer (-0.048); `fc1` separates SVHN better (+0.087 [+0.027, +0.148]). `fc2` stays the headline layer. Recorded in the Phase 2 protocol, section 5.
