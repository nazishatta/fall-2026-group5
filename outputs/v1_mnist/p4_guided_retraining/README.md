# Phase 4 results: Guided Retraining (`outputs/v1_mnist/p4_guided_retraining/`)

> `p4` in the folder name means **Phase 4**.

Retraining the CNN using what the SOM showed, and checking the result with a new SOM. Every arm is trained with 5 seeds and compared with standard training and with a random-selection control. Protocol: `src/v1_mnist/docs/p4_guided_retraining/RETRAINING_PROTOCOL.md`.

| Folder | What is inside |
|---|---|
| [`hard_example_mining/`](hard_example_mining/README.md) | The pools of hard training images, and the curriculum comparison |
| [`targeted_augmentation/`](targeted_augmentation/README.md) | Retraining with altered copies of the hard images |
| [`som_reanalysis/`](som_reanalysis/README.md) | New SOMs on the retrained CNNs, old map against new |
| [`novelty_after_retraining/`](novelty_after_retraining/README.md) | SVHN novelty scores on the retrained networks |

Git keeps only the small files here (`.json`, `.csv`, `.dvc` pointers and READMEs). Larger files are downloaded with `dvc pull`; figures are made again by the pipeline. Checkpoints, predictions, embeddings and figures are kept with DVC.
