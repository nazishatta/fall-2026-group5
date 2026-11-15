# SOM re-analysis (Phase 4)

Takes the retrained LeNet-5 networks, extracts their `fc2` embeddings again, trains a new SOM on each with the Phase 2 recipe, and compares the maps with the networks' original map. It asks: did the cluster boundaries sharpen, did the error hotspots shrink. Rules, measures and success statements: `docs/p4_guided_retraining/som_reanalysis/SOM_REANALYSIS_PROTOCOL.md` (fixed before any run). Test images are never read.

Arms re-analysed (`configs/som_reanalysis.yaml`): `standard`, `augmented_som`, `augmented_random`, seeds 1 to 5, plus the Phase 2 SOM as a read-only reference.

| File | What it does |
|---|---|
| `settings.py` | reads `som_reanalysis.yaml`, `retraining.yaml`, `som.yaml`; cell names and folders |
| `run_state.py` | SHA-256 record of the protocol, configs and checkpoints; stops if any changes |
| `cell_embeddings.py` | stage `embeddings` (needs PyTorch): `fc2` features of each checkpoint, checked against the saved predictions |
| `som_cell.py` | stage `train_som`: new SOM per cell (Phase 2 recipe), assignments, quality; stores the reference cell |
| `som_measures.py` | purity, low-purity share, class scatter index, hotspot error rate and share |
| `cka.py` | linear CKA between embeddings |
| `comparison.py` | paired differences over seeds, t-interval, statements R1 to R5 |
| `analysis.py` | stage `analyze`: tables from the stored assignments |
| `reanalysis_figures.py` | PDF figures (measures by arm, purity maps, CKA matrix) |

Runner: `pipeline/p4_guided_retraining/run_som_reanalysis.py --stage embeddings|train_som|analyze`.
Output: `outputs/v1_mnist/p4_guided_retraining/som_reanalysis/som_reanalysis_<SOM run ID>[_<revision>]/`. Tests: `tests/p4_guided_retraining/som_reanalysis/`.
