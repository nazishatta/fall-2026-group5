# `hard_example_mining` results (Fashion-MNIST)

| Run folder | What is inside |
|---|---|
| `hard_example_pools_som_15x15_nb11_ep250_s42/` | The SOM, random, max-softmax and hotspot pools (`results.json`, `tables/pool_summary.csv`, `tables/pool_by_class.csv`; per-image difficulty in `difficulty/`, DVC). The SOM pool holds 19,997 of the 49,000 training images. |
| `curriculum_comparison_som_15x15_nb11_ep250_s42/` | The curriculum arms (`standard`, `curriculum_som`, `curriculum_random`, `curriculum_msp`), 5 seeds each: `metrics/`, `tables/` (validation summary), `test_results.json` (the single test pass), `test_breakdown.json`; checkpoints, predictions, figures and logs in DVC. |

Statements C1 and C2: `src/v1_fashion_mnist/docs/p4_guided_retraining/RETRAINING_RESULTS.md`, section 2.

Git keeps the small `.json`/`.csv` results and the `.dvc` pointers; checkpoints, predictions, maps, embeddings, scores, figures and logs are kept with DVC (`dvc pull`). Results are never overwritten: a repeated step gets a new run ID.
