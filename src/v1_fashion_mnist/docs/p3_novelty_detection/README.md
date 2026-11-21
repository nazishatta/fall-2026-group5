# Documentation, p3_novelty_detection (Fashion-MNIST)

Phase 3, novelty detection: can the map flag images unlike Fashion-MNIST?

Protocol: [`NOVELTY_DETECTION_PROTOCOL.md`](NOVELTY_DETECTION_PROTOCOL.md). Changes and deviations: [`../DEVIATIONS.md`](../DEVIATIONS.md).

## How Phase 3 is run: three stages

Several steps check the results of earlier steps (a hash, a threshold, an AUROC they must reproduce). Those values exist only after the earlier step has run, so the configs in `src/v1_fashion_mnist/component/configs/` hold them as `[TBD]` and are filled in between stages, each value copied from a saved result. A step given a `[TBD]` value stops at once. Run from the code root on EC2; `F` is the Fashion-MNIST config folder.

```bash
F=src/v1_fashion_mnist/component/configs
P=src/v1_mnist/component/pipeline/p3_novelty_detection
```

| Stage | Steps | Needs |
|---|---|---|
| 1 | `qe_novelty`, `qe_ood`, `qe_random_control`; `qe_u_matrix`, `u_matrix_grid_figure`; `required_baselines`, `required_baselines_knn`, `standardized_autoencoder`; `novelty_statements` (O1 to O4) | Phase 1 and 2 results only |
| fill | the frozen QE threshold (`qe_summary.json`), the QE AUROC (`qe_ood_metrics.json`), the max-softmax, energy and kNN AUROCs (`required_baseline_metrics.json`) | stage 1 results |
| 2 | `svhn_benchmark`, `svhn_score_export`, `fair_budget_baselines`, `unified_novelty_benchmark`, `qe_norm_dependence`; `qe_threshold_sweep`; `proposal_gap_analysis`, `svhn_som_overlay` | stage 1 values |
| fill | SHA-256 of the score files of `svhn_benchmark` and `svhn_score_export`, the QE AUROC and miss count of `svhn_benchmark` | stage 2 results |
| 3 | `qe_msp_paired_comparison`, `qe_energy_knn_paired`, `qe_umatrix_random_ood`, `missed_svhn_figure`; `qe_umatrix_validation_figure` | stage 2 values |

```bash
# Stage 1
python $P/run_qe_novelty_detection.py --config-dir $F --steps qe_novelty qe_ood qe_random_control
python $P/run_u_matrix_analysis.py    --config-dir $F --steps qe_u_matrix u_matrix_grid_figure
python $P/run_novelty_benchmark.py    --config-dir $F --steps required_baselines required_baselines_knn standardized_autoencoder
python src/v1_mnist/component/p3_novelty_detection/novelty_benchmark/run_novelty_statements.py --config $F/novelty_statements.yaml

# Stage 2 (after the stage 1 values are filled in)
python $P/run_novelty_benchmark.py    --config-dir $F --steps svhn_benchmark svhn_score_export fair_budget_baselines unified_novelty_benchmark qe_norm_dependence
python $P/run_qe_novelty_detection.py --config-dir $F --steps qe_threshold_sweep
python $P/run_u_matrix_analysis.py    --config-dir $F --steps proposal_gap_analysis svhn_som_overlay

# Stage 3 (after the stage 2 values are filled in)
python $P/run_novelty_benchmark.py    --config-dir $F --steps qe_msp_paired_comparison qe_energy_knn_paired qe_umatrix_random_ood missed_svhn_figure
python $P/run_u_matrix_analysis.py    --config-dir $F --steps qe_umatrix_validation_figure
```

A runner skips any step whose results already exist, so a stage can be repeated after a failure. Each runner also takes `--dry-run`.

The two long jobs do not depend on the stages and can run afterwards (each trains several networks or maps):

```bash
python src/v1_fashion_mnist/component/pipeline/run_all.py --from held_out_class_novelty
python src/v1_fashion_mnist/component/pipeline/run_all.py --from feature_layer_ablation
```

Once every value is filled in, `python src/v1_fashion_mnist/component/pipeline/run_all.py --from novelty_detection --to held_out_class_novelty` runs the whole phase in one go; finished steps are skipped.

## What answers the protocol's statements

Statements O1 to O4, and the group proposal's RQ3 and RQ4, are judged in one place, `novelty_statements` (`outputs/v1_fashion_mnist/p3_novelty_detection/novelty_benchmark/fashion_mnist_svhn_novelty_statements_s42_v1/novelty_statements.json`), by the rules in the protocol's section 3 (amendment A2). The other steps give the further results the protocol lists (the QE flag and its rates, the threshold sweep, the controls, the classical baselines, the norm check, the figures); their own paired comparisons are supporting results.
