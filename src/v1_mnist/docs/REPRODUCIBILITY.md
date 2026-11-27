# Reproducibility statement

This page covers both datasets: MNIST (`src/v1_mnist/`) and Fashion-MNIST (`src/v1_fashion_mnist/`), which use the same step code.

## Environment
- Python 3.13 (the version CI uses).
- Every package version is pinned in `requirements.txt`, for example `torch==2.13.0`, `torchvision==0.28.0`, `nnsom==1.8.3` and `numpy==2.4.4`.
- CI (`.github/workflows/tests.yml`) installs the CPU build of PyTorch and runs the MNIST and Fashion-MNIST tests on every push and pull request.

## Settings and seeds
- All settings (grid size, run IDs, paths, hyperparameters) are in `component/configs/*.yaml`. `base.yaml` names the dataset and the output folder.
- Every seed is read from the config, and a step stops if its seed is missing. The headline SOM uses seed 42; the Phase 4 arms all use seeds 1 to 5.

## Runs are never overwritten
- Each result folder is named by its run ID. A result-producing step refuses to run if its results already exist, so a new or re-tuned run needs a new run ID.
- Each run records its config, the SHA-256 of its inputs (for example the CNN checkpoint) and the git commit.
- The protocols were written, and the Phase 4 protocol stamped with its SHA-256, before the test split was used. The test split was read once for Phases 3 and 4.
- Every change to a written rule is logged in `src/v1_mnist/docs/DEVIATIONS.md` and `src/v1_fashion_mnist/docs/DEVIATIONS.md`.

## Data and large files
- Small result files (`.json`, `.csv`) are in git. Checkpoints, embeddings, SOM weights and figures are tracked with DVC on the group's S3 remote (`s3://dats-capstone/shared/group5/dvc-cache`). Pulling them needs the group's credentials.
- Without those credentials, everything can be regenerated from scratch (Phase 1 trains the CNN from the raw data):

```bash
pip install -r requirements.txt
bash src/v1_mnist/shellscripts/run_pipeline.sh --from data_pipeline --dry-run
bash src/v1_fashion_mnist/shellscripts/run_pipeline.sh --dry-run
```

  Drop `--dry-run` to run. Use new run IDs in the configs if old results are present, because the runners refuse to overwrite them.

## Known differences when re-running
- **SOM training loop.** The SOMs were trained with the group's own loop (`train_som_with_history`), which recomputes each image's winning cell every epoch. NNSOM 1.8.x's own `som.train()` keeps the winners from the initial weights. See the DEVIATIONS rows dated 2026-11-26.
- **GPU and library versions.** On a GPU, the SVD in NNSOM's weight initialization can flip signs, so maps can differ in orientation. A fresh retrain of the grid comparison gave the same verdicts, with each grid's AUROC within 0.01 of the original (see the DEVIATIONS row dated 2026-11-21, commit `d73a43e`).
