# Phase 2: SOM Training & Visualization (`src/v1_mnist/component/p2_som_training_visualization/`)

Trains the SOM on the frozen CNN embeddings, then studies its clusters and where the CNN's mistakes fall. **Order:** `som_nnsom_training` → `cluster_analysis` → `error_pattern_analysis` → `interactive_report`. Runners in `pipeline/p2_som_training_visualization/`, tests in `tests/p2_som_training_visualization/`, results in `outputs/v1_mnist/p2_som_training_visualization/`. Each step folder except `interactive_report` also has an `*_EXPLAINED.md` in `docs/`.

## Folders

| Folder | What is inside |
|---|---|
| [`som_nnsom_training/`](som_nnsom_training/README.md) | Proposal Week 3: trains self-organizing maps (SOMs) with the NNSOM library on the CNN's fc2 embeddings (standardised with a scaler fitted on the training images), records the QE after every epoch, measures map quality (QE and topographic error) and draws the SOM figures. |
| [`cluster_analysis/`](cluster_analysis/README.md) | Proposal Week 4: finds the best-matching neuron of every image, then measures how pure each neuron is (one digit or many), its entropy, and the class-neuron intersection. |
| [`error_pattern_analysis/`](error_pattern_analysis/README.md) | Proposal Week 5: shows where on the SOM the CNN's mistakes fall: the error rate of every neuron, the error hotspots (one rule, in `utils/hotspots.py`), the most common digit confusions and the images in the hotspots. |
| [`interactive_report/`](interactive_report/README.md) | Builds one self-contained HTML page that shows the official SOM: the map, which digits each neuron holds, where the mistakes are, and sample images. |
