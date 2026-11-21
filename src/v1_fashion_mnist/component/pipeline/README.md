# Fashion-MNIST pipeline

[`run_all.py`](run_all.py) runs the shared step runners (`src/v1_mnist/component/pipeline/`) in proposal order with the Fashion-MNIST settings in [`../configs/`](../configs/). It adds nothing step-specific: the dataset and the output folder come from `../configs/base.yaml`.

From the code root (the folder that holds `src/` and `outputs/`):

```bash
# See every step without running anything
python src/v1_fashion_mnist/component/pipeline/run_all.py --from data_pipeline --dry-run

# Phases 1 and 2: split, LeNet-5, embeddings, SOM, cluster and error-pattern analysis, report
python src/v1_fashion_mnist/component/pipeline/run_all.py --from data_pipeline
```

Or use the shell script [`../../shellscripts/run_pipeline.sh`](../../shellscripts/run_pipeline.sh), which does the same.

The same runner covers every later step too (`--to STEP`; the list and the stop points before the test split are in [`src/v1_mnist/component/pipeline/README.md`](../../../v1_mnist/component/pipeline/README.md)). A step runs only once its config is in `../configs/`: a step without one is skipped with the message "not part of this dataset's study". The Phase 3 and 4 configs are added after the outside set was fixed (SVHN, `docs/p3_novelty_detection/NOVELTY_DETECTION_PROTOCOL.md`). The phase subfolders hold the Fashion-MNIST runners added for later steps.
