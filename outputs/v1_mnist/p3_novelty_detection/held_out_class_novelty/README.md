# Held-out-digit novelty results (`outputs/v1_mnist/p3_novelty_detection/held_out_class_novelty/`)

For each of 5 class splits a CNN and a SOM were trained on 6 digits; the other 4 digits are the new classes. Code: `src/v1_mnist/component/p3_novelty_detection/held_out_class_novelty/`. Protocol: `src/v1_mnist/docs/p3_novelty_detection/held_out_class_novelty/HELD_OUT_CLASS_NOVELTY_PROTOCOL.md`.

Run folder `held_out_class_novelty_lenet5_som15x15_nb11_ep250_s42/`:

| Path | What is inside |
|---|---|
| `results.json` | The verdicts across the 5 splits (QE against each other detector, with t-intervals) |
| `tables/split_scores.csv` | AUROC of every detector in every split |
| `tables/qe_minus_scores.csv` | QE minus each detector, per split and on average |
| `splits/split<k>/` | One split: its digits, metrics and run records (the trained networks and SOMs are kept with DVC) |
| `metadata/inputs.json` | The inputs and their SHA-256 fingerprints |

Main result: mean AUROC QE 0.912, max-softmax 0.944, energy 0.927, Mahalanobis 0.927, kNN 0.966. kNN beats QE in every split; QE against max-softmax is inconclusive.
