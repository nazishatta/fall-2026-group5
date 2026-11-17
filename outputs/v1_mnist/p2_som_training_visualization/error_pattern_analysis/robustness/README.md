# Error-pattern robustness (`outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/robustness/`)

Checks that the error pattern found on the 15x15 map also holds on the 10x10 and 20x20 maps, using criteria written before the check (`src/v1_mnist/docs/p2_som_training_visualization/error_pattern_analysis/ERROR_PATTERN_ROBUSTNESS_CRITERIA.md`).

| File | What it is |
|---|---|
| `som_<grid>_..._error_pattern_robustness.json` | Statistics for one map (hotspot share of mistakes, link between mixing and error) |
| `error_pattern_robustness_verdict.json` | The criteria applied to all three maps, and the verdict |
