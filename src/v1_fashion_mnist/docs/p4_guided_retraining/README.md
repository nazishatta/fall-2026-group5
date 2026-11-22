# Documentation, p4_guided_retraining (Fashion-MNIST)

Phase 4, guided retraining: does retraining on images the map picks beat picking at random?

Protocol: [`RETRAINING_PROTOCOL.md`](RETRAINING_PROTOCOL.md). Configs:
`src/v1_fashion_mnist/component/configs/` (`retraining.yaml`, `som_reanalysis.yaml`,
`som_reanalysis_msp_arm.yaml`, `novelty_after_retraining.yaml`). Code: shared with MNIST.

## Run order

Every command runs from the code root with `--config src/v1_fashion_mnist/component/configs/<config>`;
the runners are in `src/v1_mnist/component/pipeline/p4_guided_retraining/`.

1. `run_hard_example_mining.py`: the SOM, random, max-softmax and hotspot pools (training images only).
2. `run_targeted_augmentation.py --stage check`: the transform check. The Phase 1 network's validation
   accuracy on altered images must be within 2 percentage points of the originals, and the contact sheet
   (`figures/augmentation_contact_sheet.pdf`) must show recognisable clothing. If either fails, the ranges
   are made smaller under a new revision and recorded in `../DEVIATIONS.md`. The train stage refuses to
   start until the check has passed.
3. `--stage train`, then `--stage summary`, for `run_targeted_augmentation.py` and
   `run_curriculum_comparison.py` (5 seeds per arm, validation images only).
4. **Stop.** The macro-F1 guard is the larger of the two `macro_f1_guard_suggestion` values (each is 2 times
   the `standard` arm's validation macro-F1 standard deviation over the 5 seeds), found in each experiment's
   `tables/validation_summary.json`. It is copied into
   `retraining.yaml` (`evaluation.macro_f1_guard`) and into the protocol (section 2, in place of the
   `[TBD]`). A test checks that the copied value equals its rule.
5. `run_protocol_stamp.py --config .../retraining.yaml`, by hand: freezes the protocol and `retraining.yaml`.
6. `--stage final_test`, then `--stage breakdown`, for both experiments: the single test pass.
7. `run_som_reanalysis.py` (both configs) and `run_novelty_after_retraining.py`.

`run_all.py --config-dir src/v1_fashion_mnist/component/configs --from guided_retraining` runs steps 1 to 3
and stops; `--from guided_retraining_test`, asked for by name, runs steps 6 and 7 once the stamp exists.

Record the outcome in the protocol's section 5 only after step 7: the stamp records the protocol file's
SHA-256, and the test-pass stages check it.
