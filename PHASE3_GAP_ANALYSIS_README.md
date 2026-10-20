# Proposal-specific Phase 3 follow-up (additive patch)

This bundle changes **no existing repository files**. It adds Phase 3-only utilities, a new config, a runner, and tests. Do not assume the work is scientifically complete until it is applied to the latest PR branch and rerun using verified local data.

## Implemented

- Actual **MNIST-validation sample-level QE overlay**: maps mean QE of images assigned to each 15×15 BMU; empty cells remain blank. This is not the original U-Matrix heatmap and does not show SVHN.
- **Descriptive SVHN miss audit**: reports flagged/missed count, rate, and QE range from saved historical SVHN scores and the frozen training-derived threshold. It does not invent causal explanations.
- A distinct run ID, provenance hashes, explicit leakage controls and refusal to overwrite existing run IDs.
- Four narrow pytest checks and PDF-only output via shared `save_pdf`.

## Not yet complete

The professor's **"why"** requirement cannot be established merely from miss IDs and scores. It needs visual/feature-level examination and/or SVHN-to-BMU assignments, not supplied in this snapshot. This patch explicitly marks that limitation. Unusual Isolation Forest / autoencoder scores still need an appropriately isolated follow-up experiment before claims about their causes.

## On your Mac: review first, no git or modifications in this chat

1. Review the files and compare with your current PR #17 branch. The provided repository ZIP may be behind your live working tree.
2. Apply only on your own novelty branch if you approve, with no changes to Phase 1/2 or shared scripts.
3. Confirm that required DVC score arrays are restored and that the input hashes match. Do **not** overwrite any older run.
4. Run `python3 -m pytest src/v1_mnist/tests/ -q` in a working pinned environment and Ruff on the new files. Stop if tests fail; do not push.
5. Run `python3 -m src.v1_mnist.component.pipeline.p3_novelty_detection.run_proposal_gap_analysis --config src/v1_mnist/component/configs/phase3_proposal_gap_analysis.yaml` after reviewing the config. Generated PDFs must stay in gitignored outputs (DVC-track if large).

No MNIST final test is accessed, and no experiment/model fitting is performed in this patch.
