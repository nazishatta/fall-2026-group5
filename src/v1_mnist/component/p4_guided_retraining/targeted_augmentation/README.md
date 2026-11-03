# Targeted augmentation (Phase 4)

Adds altered copies of the hard training images and retrains LeNet-5, then measures the error in the SOM's error hotspots and the damage elsewhere. All arms add the same number of images, so a gain cannot come from more data alone.

Arms (`configs/retraining.yaml`, section `augmentation`): `standard`; `augmented_som`, `augmented_random`, `augmented_msp` (SOM pool, 4,217 images, times 3 copies); `augmented_hotspot`, `augmented_random_hotspot` (hotspot pool, 685 images, times 3 copies); `augmented_som_paced`, `augmented_random_paced` (same images as the SOM and random arms, but the copies come in gradually: 10% of the pool, +10% every 5 epochs, as for the curriculum). Rules and success criteria: `docs/p4_guided_retraining/RETRAINING_PROTOCOL.md`.

| File | What it does |
|---|---|
| `augmentation_arms.py` | builds each arm's training set: the original images plus the altered copies, with true labels |
| `augmentation_check.py` | contact sheet and the Phase 1 CNN's accuracy on altered validation images (the transform check) |
| `../../utils/image_transforms.py` | small rotation, shift, zoom, shear; no flips; seeded |

Runner: `pipeline/p4_guided_retraining/run_targeted_augmentation.py --stage check|train|summary|final_test`; commands in `../HOW_TO_RUN.md`.
Output: `outputs/v1_mnist/p4_guided_retraining/targeted_augmentation/augmentation_comparison_<SOM run ID>/`. Tests: `tests/p4_guided_retraining/targeted_augmentation/`.
