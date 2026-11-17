# Protocol: Optuna tuning of the SOM settings for the 10x10, 15x15 and 20x20 grids

Written: 2026-11-15, before any real (non-smoke) Optuna run. Commit this file before running.
Every run records this file's SHA-256 (in `selection.json` and `confirmation.json`), so any later change is visible.

Code, in `component/p2_som_training_visualization/som_nnsom_training/`:
`optuna_grid_tuning.py` (stage 1, the search), `optuna_confirmation.py` (stage 2, the seed check),
`optuna_selection.py` (the selection and comparison rules), `optuna_common.py` (shared inputs and training).
Config: `component/configs/som_optuna.yaml`. Tests: `tests/p2_som_training_visualization/som_nnsom_training/test_optuna_selection.py`.

## 1. Why this exists

At the 10-06 working session the instructor asked for the SOM grids to be tuned with Optuna
(https://optuna.org; Akiba et al., 2019), using Bayesian optimisation against an objective, with the
same budget for every grid. Until now each grid was tuned by a 3-point neighbourhood search
(`PREREGISTRATION_grid_comparison.md` §4: 0.20, 0.47 and 0.73 of the side length, 250 epochs), which
chose `init_neighborhood` 7, 11 and 15 for 10x10, 15x15 and 20x20.

Before starting, we wrote to the instructor that NNSOM has no learning rate and that we would tune the
starting neighbourhood size and the number of epochs on all three map sizes, with the same number of
trials for each. The instructor agreed (2026-10-25) and asked us to keep the project's own trainer.

## 2. What is tuned, and why there is no learning rate

The 10-06 note mentioned the learning rate (alpha). **NNSOM's SOM has no learning rate.** It is a *batch*
SOM: in every epoch each neuron's weight is set directly to the neighbourhood-weighted average of the inputs
it wins, so there is no step size. NNSOM's `SOM.train(x, init_neighborhood, epochs, steps)` exposes only
these settings. The trainer is the project's `train_som_with_history()`, which made the headline model
`som_15x15_nb11_ep250_s42_v2` (the instructor asked us to keep it).

| Setting | Searched values |
|---|---|
| Starting neighbourhood | a **fraction of the side length** from 0.10 to 1.00, turned into `init_neighborhood` = fraction x side, rounded half up, at least 1 |
| `epochs` | 100 to 400 in steps of 50 |
| `steps` | not tuned: equal to `epochs`, as in every project run |

Searching the neighbourhood as a fraction gives every grid the **same search space**, as the 3-point study
did. A whole-number search from 1 to the side would give 20x20 twice as many neighbourhood values as 10x10
for the same number of trials. The 3-point settings and 250 epochs are inside the space.

## 3. Stage 1: the search (the same budget for every grid)

- **20 trials per grid.** The first 8 are random; the TPE sampler (`optuna.samplers.TPESampler`, seed 42)
  chooses the other 12 from the results so far.
- **Why 20.** The space has about 70 (10x10) to 140 (20x20) distinct settings. The 3-point study used 3
  trials per grid on one setting; 20 trials on two settings is about 7 times that budget, and all three grids
  finish in about one day on EC2. No rule fixes the number; it is fixed here, before any run, and is the same
  for every grid.
- Every trial trains on all 49,000 training embeddings with SOM seed 42 (the v2 runs' seed). Two fractions
  can round to the same neighbourhood; a setting already trained in this study is **reused, not retrained**
  (same seed, same result). The trial still counts toward the 20, and `selection.json` records how many
  distinct settings were trained.
- Inputs: the frozen LeNet-5 fc2 embeddings. Their SHA-256 is checked against `configs/som.yaml`
  (`expected_inputs`) before training. MinMax scaler to [-1, 1], fitted on the training split.

## 4. Objective (validation features only)

Each trial is scored on the 10,500 validation embeddings by two numbers, both minimised: validation
**quantization error** (NNSOM definition) and validation **topographic error TE1**. They pull in opposite
directions (a small neighbourhood lowers QE and raises TE1), so the study is two-objective and keeps the
Pareto-optimal trials. TE1+2 is recorded for every trial.

No labels, no CNN correctness and no test images are used, so tuning cannot learn where the CNN makes
mistakes.

## 5. Selection rule (one setting per grid)

1. Keep the Pareto-optimal trials on (validation QE, validation TE1).
2. Among them, pick the lowest equal-weight rank sum of validation QE, TE1 and TE1+2. This is the original
   grid-selection protocol's rule, used within one grid only.
3. Ties: smaller `init_neighborhood`, then fewer epochs, then the earlier trial.

## 6. Stage 2: the seed check

Stage 1 uses one SOM seed, so a setting can win partly by luck of that seed, and its stage-1 scores are
optimistic (it was picked because they were good). Stage 2 removes both problems:

- For each grid, the **Optuna choice** and the **3-point choice** are both retrained on **5 new SOM seeds
  (1 to 5)**, never the search seed. If Optuna chose the 3-point setting, it is trained once per seed and
  reported as "same setting".
- For validation QE, TE1 and TE1+2: the paired difference (Optuna minus 3-point) per seed, its mean and a
  95% t-interval (4 degrees of freedom). Per metric: **"Optuna better"** if the interval is entirely below 0,
  **"Optuna worse"** if entirely above 0, otherwise **"no clear difference"**.
- Overall statement per grid:
  - no metric differs: **3-point search confirmed**;
  - Optuna better on at least one metric and worse on none: **Optuna setting better**;
  - the reverse: **3-point setting better**;
  - better on one and worse on another: **trade-off** (the settings balance QE and TE differently; neither
    is better overall).

## 7. What the result is used for (fixed now)

- This does **not** choose between grids. Grid sizes are not compared by QE or TE, which change with grid size
  by construction. That comparison is `GRID_COMPARISON_RESULTS.md` (15x15 and 20x20 indistinguishable at
  predicting the CNN's mistakes).
- The **headline model stays `som_15x15_nb11_ep250_s42_v2`**. Phases 2 to 4 are built on it, the instructor
  asked to keep it, and the test split has already been used; swapping it now would be a post-hoc change.
- Stage 2's overall statement is reported for each grid. If it is "Optuna setting better" for 15x15, that is
  reported as a limitation of the headline model, and any re-run of Phase 2 on the Optuna setting is
  exploratory, with its own run IDs.

## 8. Leakage control

- Training split: SOM training and scaler fitting only.
- Validation split: features only, for QE and TE. Labels and CNN predictions are not part of any score.
- Test split: loaded by the shared data loader (it checks all three splits) but never passed to Optuna or the
  seed check. Every result file records `labels_used: false` and `test_evaluated: false`.

## 9. Runs and records

- Stage 1 run ID `som_optuna_<grid>_t20_s42_v1`; stage 2 run ID `som_optuna_confirm_<grid>_seeds5_v1`.
  An existing run folder is never overwritten; a change needs a new `run.revision` in the config.
- Under `outputs/v1_mnist/p2_som_training_visualization/som_optuna_tuning/`: `trials.csv` and
  `selection.json` (stage 1), `confirmation_runs.csv` and `confirmation.json` (stage 2), `run.log` (both),
  and `optuna_grid_summary.csv` (all grids).
- `--smoke-test` runs (5x5 map, small subsets, a few short trials, run IDs ending `_smoke`) only check the code
  and are never reported.
- Expected time on EC2, run one after another: stage 1 about 6 to 7 hours, stage 2 about 3 to 4 hours.

## 10. Recorded outcome

Written 2026-11-17 from the committed result files (commit 7c78e21). Sections 1 to 9 are unchanged.
The runs recorded the SHA-256 of this file as it was before this section was filled in
(`1d287c0f8c03…`, commit 4a3d6f6); filling in this section changes the file's SHA-256, as expected.
Every result file records `labels_used: false` and `test_evaluated: false`.

**Stage 1 (search, seed 42).** Distinct settings trained out of 20 trials: 17 (10x10), 17 (15x15), 19 (20x20).

| Grid | 3-point choice | Optuna choice | Search-seed scores of the Optuna choice (QE / TE1 % / TE1+2 %) |
|---|---|---|---|
| 10x10 | nb 7, 250 epochs | nb 8, 100 epochs (trial 12) | 1.1565 / 3.37 / 1.13 |
| 15x15 | nb 11, 250 epochs | nb 15, 250 epochs (trial 9) | 1.0305 / 5.13 / 2.95 |
| 20x20 | nb 15, 250 epochs | nb 15, 200 epochs (trial 19) | 0.9622 / 6.90 / 4.50 |

**Stage 2 (seeds 1 to 5).** Mean ± SD over the 5 seeds, then the paired difference (Optuna minus 3-point)
with its 95% t-interval. Lower is better for every metric.

| Grid | Metric | 3-point setting | Optuna setting | Difference [95% interval] | Verdict |
|---|---|---|---|---|---|
| 10x10 | QE | 1.1937 ± 0.0005 | 1.1566 ± 0.0004 | −0.0371 [−0.0382, −0.0360] | Optuna better |
| 10x10 | TE1 % | 4.96 ± 0.10 | 3.41 ± 0.05 | −1.55 [−1.63, −1.47] | Optuna better |
| 10x10 | TE1+2 % | 1.85 ± 0.02 | 1.10 ± 0.04 | −0.75 [−0.78, −0.72] | Optuna better |
| 15x15 | QE | 1.0429 ± 0.0011 | 1.0479 ± 0.0066 | +0.0050 [−0.0025, +0.0126] | no clear difference |
| 15x15 | TE1 % | 6.35 ± 0.11 | 6.21 ± 0.47 | −0.14 [−0.81, +0.53] | no clear difference |
| 15x15 | TE1+2 % | 3.46 ± 0.05 | 3.45 ± 0.31 | −0.01 [−0.43, +0.42] | no clear difference |
| 20x20 | QE | 0.9696 ± 0.0007 | 0.9661 ± 0.0047 | −0.0035 [−0.0095, +0.0025] | no clear difference |
| 20x20 | TE1 % | 6.37 ± 0.26 | 7.01 ± 0.28 | +0.64 [+0.03, +1.26] | Optuna worse |
| 20x20 | TE1+2 % | 4.16 ± 0.27 | 4.69 ± 0.17 | +0.53 [+0.14, +0.92] | Optuna worse |

**Overall (section 6 rule).**

| Grid | Overall |
|---|---|
| 10x10 | **Optuna setting better** (nb 8, 100 epochs) |
| 15x15 | **3-point search confirmed: no clear difference** |
| 20x20 | **3-point setting better** |

**What this means (section 7 rule, fixed before the runs).**

- **15x15, the headline grid:** the setting used for every Phase 2 to 4 result (`som_15x15_nb11_ep250_s42_v2`)
  is confirmed. The Optuna choice looked better on the search seed (TE1 5.13% against 6.18% for the v2 model)
  but not on new seeds, where it was also less stable (TE1 SD 0.47 against 0.11).
- **20x20:** the 3-point setting is better on topographic error. The Optuna choice won on the search seed only.
  This is why stage 2 exists: a setting picked for its scores on one seed looks better than it is.
- **10x10:** Optuna found a better setting, on all three metrics and with fewer epochs. 10x10 is used only as a
  robustness map for the Phase 2 error-geography result, which held on it (`ERROR_PATTERN_ROBUSTNESS_RESULTS.md`).
  This is reported as a limitation; the robustness check is not re-run, because the headline result does not
  depend on it.
- This does not compare grid sizes (section 7). The headline grid and model stay as they are.
