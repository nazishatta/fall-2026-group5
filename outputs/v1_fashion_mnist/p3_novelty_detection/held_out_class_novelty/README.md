# `held_out_class_novelty` results (Fashion-MNIST)

Near novelty: for each of 5 class splits a LeNet-5 and a SOM were trained on 6 classes; the other 4 classes are the new ones. The splits are MNIST's class numbers (drawn with seed 2026): [0, 1, 5, 6], [2, 3, 5, 8], [1, 5, 6, 7], [2, 5, 6, 8], [0, 4, 5, 8]. Code: `src/v1_mnist/component/p3_novelty_detection/held_out_class_novelty/`. Protocol: `src/v1_mnist/docs/p3_novelty_detection/held_out_class_novelty/HELD_OUT_CLASS_NOVELTY_PROTOCOL.md`, with Fashion-MNIST classes (Fashion-MNIST Phase 3 protocol, section 4).

Run folder `held_out_class_novelty_lenet5_som15x15_nb11_ep250_s42/`:

| Path | What is inside |
|---|---|
| `results.json` | The verdicts H1 to H4 across the 5 splits (QE against each other detector, with t-intervals) |
| `tables/split_scores.csv` | AUROC of every detector in every split |
| `tables/qe_minus_scores.csv` | QE minus each detector, per split, with intervals |
| `splits/split<k>/` | One split: `metrics.json`; the network, map, embeddings and scores (DVC) |
| `metadata/inputs.json` | The inputs and their SHA-256 fingerprints, including the config's |
| `logs/` (+ `logs.dvc`) | Run logs (DVC) |

Main result: mean AUROC QE 0.794, max-softmax 0.749, energy 0.783, Mahalanobis 0.822, nearest neighbours 0.842. QE beats max-softmax in every split (H2 shown); nearest neighbours beat QE in every split. Note: some keys in `results.json` carry the MNIST code's names (`mnist_test_read`, `..._digits`); in this run they refer to the Fashion-MNIST test split and classes (`src/v1_fashion_mnist/docs/DEVIATIONS.md`).
