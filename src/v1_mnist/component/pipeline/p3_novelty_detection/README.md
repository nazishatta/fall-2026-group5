# Phase 3 runners (`src/v1_mnist/component/pipeline/p3_novelty_detection/`)

One runner per Phase 3 step, named after the step folder it runs (`run_<step>.py` runs `component/p3_novelty_detection/<step>/`). Run from the code root (the folder that holds `src/` and `outputs/`). Each runner reads its settings from `component/configs/`, checks that its run ID is new, and never overwrites a finished run. See `component/pipeline/README.md` for which runners `run_all.py` calls and which are run on their own.

## Files

| File | What it is |
|---|---|
| `run_held_out_class_novelty.py` ▶ | Novelty detection on held-out classes (Phase 3, near novelty): train on 6 digits, treat the other 4 as novel. |
| `run_novelty_benchmark.py` ▶ | Novelty benchmark: SVHN comparison against the required baselines. |
| `run_qe_novelty_detection.py` ▶ | QE novelty detection: quantization-error scoring and external OOD evaluation. |
| `run_u_matrix_analysis.py` ▶ | U-Matrix analysis of the frozen SOM as a novelty signal. |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
