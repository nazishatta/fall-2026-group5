# `p1_model_training_baseline` results (Fashion-MNIST)

Phase 1, model training baseline: load and split Fashion-MNIST, train LeNet-5, save the embeddings.

```
p1_model_training_baseline/
├── data_pipeline/             the 70/15/15 split, its metrics and figures (DVC)
└── cnn_baseline_training/     the LeNet-5 network (checkpoints), embeddings, logs and figures (DVC); metrics
```

Protocol and recorded results: `src/v1_fashion_mnist/docs/p1_model_training_baseline/MODEL_TRAINING_PROTOCOL.md`.
Each folder's README names the code and config that wrote it.

Git keeps only the small files here (`.json`, `.csv`, `.dvc` pointers and READMEs). Networks, embeddings,
figures and logs are kept with DVC: run `dvc pull` to get them.
