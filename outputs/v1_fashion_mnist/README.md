# Fashion-MNIST results

Results of the Fashion-MNIST runs, in the same phase and step folders as `outputs/v1_mnist/`. Git keeps only small `.json`/`.csv` results, `.dvc` pointers and READMEs; large files (networks, maps, embeddings, score arrays, figures) are kept with DVC (S3): run `dvc pull` to get them. Protocols, one per phase: `src/v1_fashion_mnist/docs/`.

| Folder | Phase | Status |
|---|---|---|
| [`p1_model_training_baseline/`](p1_model_training_baseline/) | 1: data split and the LeNet-5 baseline | run |
| [`p2_som_training_visualization/`](p2_som_training_visualization/) | 2: SOM training, clusters and error patterns | run |
| [`p3_novelty_detection/`](p3_novelty_detection/README.md) | 3: novelty detection (SVHN and held-out classes) | run |
| [`p4_guided_retraining/`](p4_guided_retraining/) | 4: SOM-guided retraining | not run yet |
