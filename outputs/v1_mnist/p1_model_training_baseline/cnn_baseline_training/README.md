# CNN Baseline Results (Week 2: Train Baseline CNN)

Week 1 outputs (class-distribution charts, sample images, split sizes) are in `../data_pipeline/`.
The Week 1 log `prepare_data.log` stays in `logs/` here because DVC tracks the three Phase 1 logs as one folder.

## Run IDs

Each CNN training run has a run ID (`run.id` in `src/v1_mnist/component/configs/cnn_baseline.yaml`).
Its checkpoint is `checkpoints/<run.id>.pth`; its test metrics, run record (`metrics/<run.id>_run.json`:
seed, config and checkpoint fingerprints, library versions), figures and logs start with `<run.id>_`.
The scripts refuse to overwrite an existing run, so a new or re-tuned run needs a new `run.id`
and a new `paths.embeddings_dir`.

The run every SOM was built from is `lenet5_best`. It was made before run IDs existed, so only its
checkpoint carries the ID; `metrics/test_evaluation.json`, the figures, the logs and `embeddings/`
keep their old names. Its seed (42) is in the config.
