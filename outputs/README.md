# Outputs (`outputs/`)

Results written by the code, one folder per dataset version. The folder names match the code folders in `src/`.

| Folder | What is inside |
|---|---|
| [`v1_mnist/`](v1_mnist/README.md) | Every result of the MNIST study, by phase and step |

Git keeps only the small files here (`.json`, `.csv`, `.dvc` pointers and READMEs). Larger files are downloaded with `dvc pull`; figures are made again by the pipeline. The rules are in the root `.gitignore`, and `src/v1_mnist/tests/test_repository_rules.py` checks them.
