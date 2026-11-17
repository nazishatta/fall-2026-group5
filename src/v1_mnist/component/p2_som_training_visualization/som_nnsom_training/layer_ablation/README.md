# Feature-layer ablation (`src/v1_mnist/component/p2_som_training_visualization/som_nnsom_training/layer_ablation/`)

SOMs on two layers of the frozen Phase 1 LeNet-5, `fc1` (120 numbers) and `fc2` (84 numbers, the layer every other step uses), with the same Phase 2 recipe and 5 SOM seeds each. They are compared on the error-cell AUROC, the purity and the QE novelty AUROC against SVHN. The proposal's paper outline asks for this ablation. The headline layer stays `fc2` whatever the result.

Runner: `pipeline/p2_som_training_visualization/run_feature_layer_ablation.py` (`--stage extract`, then `som`, then `compare`). Config: `configs/feature_layer_ablation.yaml`. Protocol: `docs/p2_som_training_visualization/som_nnsom_training/FEATURE_LAYER_ABLATION_PROTOCOL.md`. Results: `outputs/v1_mnist/p2_som_training_visualization/feature_layer_ablation/`.

## Files

| File | What it is |
|---|---|
| `comparison.py` | Stage 3: fc1 - fc2 per SOM seed, t-intervals and the three-way verdicts of L1 to L3 (protocol section 4). |
| `extraction.py` | Stage 1: fc1 and fc2 embeddings of the Phase 1 LeNet-5 for training, validation and SVHN test images. |
| `measures.py` | Measures A1 to A3 of one map, and the reported extras (protocol section 3). |
| `provenance.py` | Provenance of a feature-layer ablation run: SHA-256 of the protocol, the configs and the frozen inputs. |
| `settings.py` | Settings of the feature-layer ablation: feature_layer_ablation.yaml plus the configs it names. |
| `som_runs.py` | Stage 2: one SOM per layer and SOM seed (Phase 2 recipe), scored with A1 to A3. |
