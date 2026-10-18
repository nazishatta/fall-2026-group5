# error_pattern_baselines (Phase 2: SOM training and visualization)

Checks whether the SOM's error-geography result is special: k-means with as many clusters as the SOM has neurons, kNN impurity, and the CNN's max-softmax probability are scored with the same rules on the same validation images.
Plan and deviations: `src/v1_mnist/docs/p2_som_training_visualization/error_pattern_analysis/baselines/PLAN.md`.

| File | Responsibility |
|---|---|
| `settings.py` | Reads this check's config and the headline SOM config |
| `inputs.py` | Verifies the SOM model and embeddings (SHA-256), loads validation/training data and the SOM's cells |
| `partitions.py` | k-means cells, kNN impurity (the max-softmax score is in `utils/scores.py`) |
| `hotspot_comparison.py` | Hotspot concentration for any partition, using the project's own statistics; reproduction check |
| `detection.py` | AUROC and average precision over repeated half-splits; bootstrap intervals |
| `result_tables.py` | Summary CSV and the LaTeX table for the report |
| `baseline_stages.py` | The three stages (`som`, `kmeans`, `final`) |
| `run_preflight.py` | Refuses to overwrite a finished run |

Run it with `src/v1_mnist/component/pipeline/p2_som_training_visualization/run_error_pattern_baselines.py` (see its docstring).
Results go to `outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/error_pattern_baselines/<run_id>/` (for the 15x15 map: `error_pattern_baselines_som_15x15`).
