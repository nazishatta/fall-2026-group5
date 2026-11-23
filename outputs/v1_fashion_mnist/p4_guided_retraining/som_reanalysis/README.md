# `som_reanalysis` results (Fashion-MNIST)

| Run folder | What is inside |
|---|---|
| `som_reanalysis_som_15x15_nb11_ep250_s42/` | New 15x15 SOMs on the `standard`, `augmented_som` and `augmented_random` networks (seeds 1-5), validation images only: `tables/statements.json` (R1 to R5), `tables/paired_comparisons.json`, `tables/cell_measures.json`, `tables/cka.json`, `metrics/`, `metadata/`; maps, embeddings, assignments and figures in DVC. |
| `som_reanalysis_msp_arm_som_15x15_nb11_ep250_s42/` | The secondary run with the max-softmax arm (statements S1 and S2), same files. |

Statements and their reading: `src/v1_fashion_mnist/docs/p4_guided_retraining/RETRAINING_RESULTS.md`, section 3.

Git keeps the small `.json`/`.csv` results and the `.dvc` pointers; checkpoints, predictions, maps, embeddings, scores, figures and logs are kept with DVC (`dvc pull`). Results are never overwritten: a repeated step gets a new run ID.
