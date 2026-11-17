# Held-out-digit novelty test (`src/v1_mnist/component/p3_novelty_detection/held_out_class_novelty/`)

A harder ("near") novelty test, asked for in Review #5. For each of 5 class splits, a new CNN and SOM are trained on 6 digits only; the other 4 digits are the new, unseen classes. QE is compared with max-softmax, energy, Mahalanobis and kNN distance on the same embeddings. Result: QE detects the new digits (mean AUROC 0.912) but kNN does better in every split (0.966), and the comparison with max-softmax (0.944) is inconclusive.

Runner: `pipeline/p3_novelty_detection/run_held_out_class_novelty.py` (`--stage run` trains the 5 splits, then `--stage compare` makes the verdicts). Config: `configs/held_out_class_novelty.yaml`. Protocol: `docs/p3_novelty_detection/held_out_class_novelty/HELD_OUT_CLASS_NOVELTY_PROTOCOL.md`. Results: `outputs/v1_mnist/p3_novelty_detection/held_out_class_novelty/`.

## Files

| File | What it is |
|---|---|
| `class_split.py` | Known and unknown digits of one split (protocol sections 2 and 3). |
| `comparison.py` | Stage 2: statements H1 to H4 over the five splits (protocol section 7), results.json and tables. |
| `provenance.py` | Provenance of a held-out-class run: SHA-256 of the protocol, the configs and the frozen split's files. |
| `settings.py` | Settings of the held-out-class novelty step: held_out_class_novelty.yaml plus the configs it names. |
| `split_metrics.py` | Metrics of one split from its score arrays (protocol sections 5 and 6). |
| `split_run.py` | Stage 1: one split end to end (protocol sections 3 to 6). |
