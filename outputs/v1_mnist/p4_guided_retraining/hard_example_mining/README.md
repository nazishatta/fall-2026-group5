# Hard example mining results (`outputs/v1_mnist/p4_guided_retraining/hard_example_mining/`)

Results of Phase 4, Week 9 (code: `src/v1_mnist/component/p4_guided_retraining/hard_example_mining/`).

| Run folder | What is inside |
|---|---|
| `hard_example_pools_som_15x15_nb11_ep250_s42_v2/` | Which training images are hard (in a hotspot, in a low-purity neuron, or far from their neuron), and the equal-size SOM, max-softmax and random pools: `results.json`, `tables/pool_summary.csv`, `tables/pool_by_class.csv`, `difficulty/train_difficulty.csv` (DVC) |
| `curriculum_comparison_som_15x15_nb11_ep250_s42_v2/` | Standard training against curricula that let the hard images in slowly (SOM, max-softmax and random pools), 5 seeds each: `test_results.json`, `test_breakdown.json`, `metrics/`, `tables/`, and `checkpoints/`, `predictions/`, `figures/` (DVC) |

Main result: the curricula made the CNN worse, not better.
