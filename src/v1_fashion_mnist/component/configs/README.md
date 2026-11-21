# Fashion-MNIST settings

YAML settings for every Fashion-MNIST step. The config loader merges `base.yaml` from this folder into each config, so every result goes under `outputs/v1_fashion_mnist/`.

## Phase 3

One config per step of the Phase 3 protocol (`docs/p3_novelty_detection/NOVELTY_DETECTION_PROTOCOL.md`), each made from the MNIST config of the same name with only the dataset-specific values changed. `phase3_shared_provenance.yaml` holds the frozen Phase 3 inputs and their SHA-256; every Phase 3 script checks its inputs against it. `novelty_statements.yaml` judges the group proposal's RQ3 and RQ4 and this protocol's statements O1 to O4 in one run. MNIST steps that only re-created or repaired MNIST's own earlier runs have no config here, so the runners skip them. Values marked `[TBD]` are filled in from earlier steps' results, in the stages described in `docs/p3_novelty_detection/README.md`.

