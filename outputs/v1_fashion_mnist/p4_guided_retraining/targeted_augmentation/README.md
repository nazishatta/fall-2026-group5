# `targeted_augmentation` results (Fashion-MNIST)

| Run folder | What is inside |
|---|---|
| `augmentation_comparison_som_15x15_nb11_ep250_s42/` | Revision r1: the transform check with MNIST's ranges, which failed (accuracy drop 0.081, limit 0.02); `check/` and the contact sheet only. Kept as the record. |
| `augmentation_comparison_som_15x15_nb11_ep250_s42_r2/` | Revision r2: ranges halved; the check failed again (drop 0.029). `check/` and the contact sheet only. |
| `augmentation_comparison_som_15x15_nb11_ep250_s42_r3/` | Revision r3: ranges halved twice; the check passed (drop 0.012). All 8 arms, 5 seeds each: `metrics/`, `tables/` (validation summary, from which the macro-F1 guard was measured), `test_results.json` (the single test pass), `test_breakdown.json`; checkpoints, predictions, figures and logs in DVC. |

The halving rule and each check's numbers: `src/v1_fashion_mnist/docs/DEVIATIONS.md`. Statements A1 to A3: `src/v1_fashion_mnist/docs/p4_guided_retraining/RETRAINING_RESULTS.md`, section 2.

Git keeps the small `.json`/`.csv` results and the `.dvc` pointers; checkpoints, predictions, maps, embeddings, scores, figures and logs are kept with DVC (`dvc pull`). Results are never overwritten: a repeated step gets a new run ID.
