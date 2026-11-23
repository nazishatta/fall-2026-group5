# `novelty_after_retraining` results (Fashion-MNIST)

| Run folder | What is inside |
|---|---|
| `novelty_after_retraining_som_15x15_nb11_ep250_s42/` | Phase 3's novelty scores (QE, max-softmax, energy, Mahalanobis, embedding length) on the 15 retrained networks and the baseline, Fashion-MNIST validation against SVHN test: `results.json` (statements N1 to N4), `metrics/` (one file per network), `tables/`, `metadata/`; score arrays and logs in DVC. |

Statements and their reading: `src/v1_fashion_mnist/docs/p4_guided_retraining/RETRAINING_RESULTS.md`, section 4.

Git keeps the small `.json`/`.csv` results and the `.dvc` pointers; checkpoints, predictions, maps, embeddings, scores, figures and logs are kept with DVC (`dvc pull`). Results are never overwritten: a repeated step gets a new run ID.
