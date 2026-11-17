# Novelty detection after retraining (`src/v1_mnist/component/p4_guided_retraining/novelty_after_retraining/`)

Checks whether the SOM still detects SVHN images after the CNN has been retrained in Phase 4. For every retrained network (5 seeds per arm) it scores SVHN against MNIST validation with QE and the other detectors, and compares the retrained arms with standard training. Result: the SOM still detects SVHN, but no better than before.

Runner: `pipeline/p4_guided_retraining/run_novelty_after_retraining.py`. Config: `configs/novelty_after_retraining.yaml`. Protocol: `docs/p4_guided_retraining/novelty_after_retraining/NOVELTY_AFTER_RETRAINING_PROTOCOL.md`. Results: `outputs/v1_mnist/p4_guided_retraining/novelty_after_retraining/`.

## Files

| File | What it is |
|---|---|
| `cell_scoring.py` | Stage 1: score one cell (protocol sections 2 to 4). |
| `comparison.py` | Stage 2: compare the arms and decide statements N1 to N4 (protocol section 5). |
| `provenance.py` | Provenance of a novelty-after-retraining run: SHA-256 of the protocol, configs and every input. |
| `scoring.py` | The five novelty scores of one cell and their metrics (protocol sections 3 and 4). |
| `settings.py` | Settings of the novelty-after-retraining step: novelty_after_retraining.yaml plus the configs it names. |
