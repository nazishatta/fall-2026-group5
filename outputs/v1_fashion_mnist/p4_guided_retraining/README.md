# `p4_guided_retraining` results (Fashion-MNIST)

Phase 4, guided retraining: does retraining on images the map picks beat picking at random? Every arm is
trained with seeds 1-5. Protocol: `src/v1_fashion_mnist/docs/p4_guided_retraining/RETRAINING_PROTOCOL.md`
(stamped 2026-11-22 before the test split was read). Results and their reading: `src/v1_fashion_mnist/docs/p4_guided_retraining/RETRAINING_RESULTS.md`.

```
p4_guided_retraining/
├── hard_example_mining/       the hard-image pools and the curriculum comparison
├── targeted_augmentation/     retraining with altered copies of the pool images (transform checks r1, r2, r3)
├── som_reanalysis/            new SOMs on the retrained networks (main run and the max-softmax run)
└── novelty_after_retraining/  SVHN novelty scores on the retrained networks
```

Git keeps the small `.json`/`.csv` results and the `.dvc` pointers; checkpoints, predictions, maps, embeddings, scores, figures and logs are kept with DVC (`dvc pull`). Results are never overwritten: a repeated step gets a new run ID.
