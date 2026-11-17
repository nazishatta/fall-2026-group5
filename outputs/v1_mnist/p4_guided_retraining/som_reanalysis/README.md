# SOM re-analysis results (`outputs/v1_mnist/p4_guided_retraining/som_reanalysis/`)

Results of Phase 4, Week 11 (code: `src/v1_mnist/component/p4_guided_retraining/som_reanalysis/`). A new SOM is trained on the embeddings of every retrained CNN and compared with the original map. Protocol: `src/v1_mnist/docs/p4_guided_retraining/som_reanalysis/SOM_REANALYSIS_PROTOCOL.md`.

| Run folder | What is inside |
|---|---|
| `som_reanalysis_som_15x15_nb11_ep250_s42_v2/` | The primary run (standard, augmented SOM and augmented random arms, 5 seeds each): `metrics/`, `tables/` (map measures and paired comparisons), `metadata/`, and `embeddings/`, `som_models/`, `assignments/`, `figures/` (DVC) |
| `som_reanalysis_msp_arm_som_15x15_nb11_ep250_s42_v2/` | The secondary run that adds the max-softmax (least-confident) arm (`MSP_ARM_ADDENDUM.md`): same files as the primary run. Its `statements.json` names R3 and R4 compare with `augmented_msp`, not random augmentation. |

Main result (primary run): the new maps have smaller weak regions but not purer cells. Secondary run: the least-confident arm's maps show no clear hotspot drop, and SOM guidance is not clearly better than confidence guidance (5 seeds).
