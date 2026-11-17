# Hard example mining (Phase 4)

Two parts. **Pools** decide which training images are hard, before anything is retrained (no CNN is trained, no test split is loaded). **Curriculum comparison** trains LeNet-5 with standard training and with curricula that let the hard pool in gradually.

A training image is in the **SOM pool** if any of these is true (rules fixed in `docs/p4_guided_retraining/RETRAINING_PROTOCOL.md`, numbers in `configs/retraining.yaml`):

1. its neuron is one of the validation error hotspots (the project's one hotspot rule, `utils/hotspots.py`);
2. its neuron's training purity is below `pools.purity_cutoff`;
3. its distance to its neuron is above the `pools.qe_percentile` of the training distances.

Two same-size pools are built for the fair comparison: **msp** (the images the CNN is least sure about) and **random** (same digit mix as the SOM pool, drawn per seed later, at retraining time).

| File | What it does |
|---|---|
| `settings.py` | reads `retraining.yaml` and the headline SOM config |
| `inputs.py` | checks the SOM and embedding hashes, loads the training table, checks it against the saved sample IDs |
| `difficulty.py` | the three rules |
| `pool_selection.py` | msp and random pools |
| `pool_outputs.py` | per-image difficulty table, `pool_summary.csv`, `pool_by_class.csv` |
| `build_pools.py` | runs the step, refuses to overwrite a finished run, writes `results.json` with every input SHA-256 |
| `curriculum.py` | the schedule (10% of the pool, +10% every 5 epochs) and the easiest-to-hardest order |
| `curriculum_arms.py` | the arms: `standard`, `curriculum_som`, `curriculum_random`, `curriculum_msp` |

Runners: `run_hard_example_mining.py` (pools) and `run_curriculum_comparison.py --stage train|summary|final_test`; commands in `../HOW_TO_RUN.md`.
Output: `outputs/v1_mnist/p4_guided_retraining/hard_example_mining/hard_example_pools_<SOM run ID>/` and `curriculum_comparison_<SOM run ID>/`. Tests: `tests/p4_guided_retraining/hard_example_mining/`.
