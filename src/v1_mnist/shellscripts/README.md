# Shell scripts (MNIST)

| Script | What it does |
|---|---|
| [`run_pipeline.sh`](run_pipeline.sh) | Runs the MNIST pipeline: `src/v1_mnist/component/pipeline/run_all.py` from the code root, with every option passed on. Unless `--from` is given, it starts at `som_nnsom_training`. |

Example: `bash src/v1_mnist/shellscripts/run_pipeline.sh --dry-run` lists every step without running it.
