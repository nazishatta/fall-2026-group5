# Targeted augmentation results (`outputs/v1_mnist/p4_guided_retraining/targeted_augmentation/`)

Results of Phase 4, Targeted Data Augmentation (code: `src/v1_mnist/component/p4_guided_retraining/targeted_augmentation/`).

| Run folder | What is inside |
|---|---|
| `augmentation_comparison_som_15x15_nb11_ep250_s42_v2/` | First attempt, stopped before training: with shifts of up to 2 pixels the check in `check/augmentation_check.json` failed (validation accuracy on the altered images fell from 99.12% to 95.83%). Kept as the record of that check. |
| `augmentation_comparison_som_15x15_nb11_ep250_s42_v2_r2/` | The run used in the reports, with shifts of up to 1 pixel (check passed, 98.21%). Arms, 5 seeds each: `standard`, `augmented_som`, `augmented_msp` (least confident images), `augmented_random`, `augmented_hotspot`, `augmented_random_hotspot`, and the paced versions `augmented_som_paced` and `augmented_random_paced`. `test_results.json`, `test_breakdown.json` (hotspot error and error elsewhere), `metrics/`, `tables/`, and `checkpoints/`, `predictions/`, `figures/` (DVC) |

Main result: SOM-chosen hard images with altered copies cut the error in the hotspot cells from 11.4% to 7.8% without raising the error elsewhere; the max-softmax arm did about as well.
