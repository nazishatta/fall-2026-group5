# `p3_novelty_detection` results (Fashion-MNIST)

Phase 3, novelty detection: can the map flag images unlike Fashion-MNIST?

```
p3_novelty_detection/
├── qe_novelty_detection/          # quantization-error novelty scores and the frozen QE threshold
├── u_matrix_analysis/             # U-Matrix novelty scores and figures
├── novelty_benchmark/             # SVHN: the SOM against the other novelty methods; the O1 to O4 verdicts
├── held_out_class_novelty/        # near novelty: networks trained on 6 classes, the other 4 are novel
└── run_manifest.csv               # one row per run folder: the code and config that wrote it
```

Protocol: `src/v1_fashion_mnist/docs/p3_novelty_detection/NOVELTY_DETECTION_PROTOCOL.md`. The run order (three
stages, then the held-out classes) is in `src/v1_fashion_mnist/docs/p3_novelty_detection/README.md`. The
statements O1 to O4 and the group proposal's RQ3 and RQ4 are judged in
`novelty_benchmark/fashion_mnist_svhn_novelty_statements_s42_v1/novelty_statements.json`; H1 to H3 in
`held_out_class_novelty/held_out_class_novelty_lenet5_som15x15_nb11_ep250_s42/results.json`.

## Which code wrote which folder

Every run folder is listed in [`run_manifest.csv`](run_manifest.csv), one row per folder, with the same
columns as MNIST's (`outputs/v1_mnist/p3_novelty_detection/run_manifest.csv`, whose README explains them).
A test in `src/v1_mnist/tests/test_repository_rules.py` checks, for both datasets, that every folder has
exactly one row, that every script and config named exists, and that each recorded config hash still matches
today's config file. All Fashion-MNIST runs are first runs of the current code, so every row has the role
`result`; `qe_novelty` and `qe_ood` do not record their config hash (as on MNIST).

Score arrays (`.npy`, `.npz`), networks, maps and PDFs are kept with DVC; the `.json` and `.csv` results are in
git. Results are never overwritten: a repeated step gets a new run ID.
