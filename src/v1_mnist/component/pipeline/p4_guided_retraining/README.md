# Phase 4 runners (`src/v1_mnist/component/pipeline/p4_guided_retraining/`)

One runner per Phase 4 step, named after the step folder it runs (`run_<step>.py` runs `component/p4_guided_retraining/<step>/`). Run from the code root (the folder that holds `src/` and `outputs/`). Each runner reads its settings from `component/configs/`, checks that its run ID is new, and never overwrites a finished run. See `component/pipeline/README.md` for which runners `run_all.py` calls and which are run on their own.

## Files

| File | What it is |
|---|---|
| `run_curriculum_comparison.py` ▶ | Curriculum comparison (Phase 4, hard example mining): standard training against curricula. |
| `run_hard_example_mining.py` ▶ | Hard example mining (Phase 4): which training images are hard, before any retraining. |
| `run_novelty_after_retraining.py` ▶ | Novelty detection after retraining (Phase 4): Phase 3's novelty scores on the retrained networks. |
| `run_protocol_stamp.py` ▶ | Freeze the retraining protocol (run once, after every open item in the protocol is filled). |
| `run_som_reanalysis.py` ▶ | SOM re-analysis (Phase 4): re-train the SOM on the retrained CNNs' embeddings and compare the maps. |
| `run_targeted_augmentation.py` ▶ | Targeted augmentation (Phase 4): add altered copies of the pool images and retrain. |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
