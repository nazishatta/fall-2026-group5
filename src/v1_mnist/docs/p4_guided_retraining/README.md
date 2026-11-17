# Documents: Phase 4, guided retraining (`src/v1_mnist/docs/p4_guided_retraining/`)

| File or folder | What it is |
|---|---|
| `RETRAINING_PROTOCOL.md` | The protocol for the retraining experiment: which images are hard, the arms, the measures and the success criteria. Frozen on 2026-11-02 (see the stamp); never edited after the stamp. |
| `retraining_protocol_stamp.txt` | The SHA-256 fingerprint of the protocol and of `configs/retraining.yaml`, written when the protocol was frozen. |
| [`som_reanalysis/`](som_reanalysis/README.md) | `SOM_REANALYSIS_PROTOCOL.md` (measures and success statements fixed before the run), `FEEDBACK_LOOP.md` (the analyse-retrain-analyse loop and its results) and `MSP_ARM_ADDENDUM.md` (the extra run that adds the max-softmax arm). |
| [`novelty_after_retraining/`](novelty_after_retraining/README.md) | `NOVELTY_AFTER_RETRAINING_PROTOCOL.md`: does the SOM still detect SVHN after retraining? |
