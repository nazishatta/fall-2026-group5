# Error pattern analysis (`src/v1_mnist/component/p2_som_training_visualization/error_pattern_analysis/`)

Proposal Week 5. Shows where on the SOM the CNN's mistakes fall: the error rate of every neuron, the error hotspots (one rule, in `utils/hotspots.py`), the most common digit confusions and the images in the hotspots. `error_pattern_robustness*.py` repeat the analysis on the 10x10 and 20x20 maps and apply the criteria fixed beforehand. `baselines/` compares the SOM's error cells with simpler ways of finding mistakes (see its README).

Runners: `pipeline/p2_som_training_visualization/run_error_pattern_analysis.py`, `run_error_pattern_baselines.py`, `run_error_detector_budget.py`. Results: `outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/`.

## Folders

| Folder | What is inside |
|---|---|
| [`baselines/`](baselines/README.md) | Checks whether the SOM's error-geography result is special: k-means with as many clusters as the SOM has neurons, kNN impurity, and the CNN's max-softmax probability are scored with the same rules on the same validation images. |

## Files

| File | What it is |
|---|---|
| `error_figures.py` ▶ | Week 5 figures for the selected SOM (Error Pattern Analysis). |
| `error_heatmaps.py` ▶ | Week 5 heatmaps of the SOM (Error Pattern Analysis). |
| `error_pattern_robustness.py` ▶ | Error-pattern (error-geography) statistics for one SOM, judged against "errors placed at random". |
| `error_pattern_robustness_verdict.py` ▶ | Apply the pre-specified error-pattern rules (docs/p2_som_training_visualization/error_pattern_analysis/ERROR_PATTERN_ROBUSTNESS_CRITERIA.md) to the *_error_pattern_robustness.json files: headline model first, then each robustness model. |
| `error_patterns.py` ▶ | Week 5: Error Pattern Analysis. |
| `hotspot_samples.py` ▶ | Generate representative MNIST images for Week 5 SOM error hotspots. |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
