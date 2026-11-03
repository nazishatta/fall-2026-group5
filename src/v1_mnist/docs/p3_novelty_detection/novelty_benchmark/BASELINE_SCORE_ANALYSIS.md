# Why Isolation Forest and the autoencoder look odd on SVHN

Recorded: 2026-11-03 (controlled check added the same day). Development results only (MNIST validation vs SVHN test).
The official MNIST test split was not used. No old run, config or score was changed.

Two baseline results look strange: Isolation Forest scores below chance, and the
autoencoder has given very different numbers in different runs. This note puts
the facts in one place, says what is known, and says what is still only a guess.

## 1. The numbers

AUROC, MNIST validation (not novel) vs SVHN test (novel). 0.5 is chance.

| Run | Method | Setup | AUROC |
|---|---|---|---:|
| `mnist_svhn_five_method_comparison_s42_initial` | Isolation Forest | scaled input, one seed | 0.238 |
| `mnist_svhn_fair_budget_baselines_s42_46_v1` | Isolation Forest | scaled input, 10,000 training embeddings, 5 seeds | 0.280 ± 0.085 (per seed 0.14 to 0.36) |
| `mnist_svhn_unified_nine_methods_s42_46_v1` | Isolation Forest | same as fair-budget | 0.280 ± 0.085 (identical) |
| `mnist_svhn_five_method_comparison_s42_initial` | Autoencoder | raw (unscaled) input, one seed | 0.008 |
| `mnist_svhn_fair_budget_baselines_s42_46_v1` | Autoencoder | raw input, 10,000 training embeddings, 5 seeds | 0.006 ± 0.001 |
| `mnist_svhn_ae_train_standardized_s42_v1` | Autoencoder | standardized input, scaler fitted on all 49,000 training embeddings, one seed | 0.760 |
| `mnist_svhn_unified_nine_methods_s42_46_v1` | Autoencoder | standardized input, scaler fitted on the 10,000-embedding subset, 5 seeds | 0.979 ± 0.017 |

Isolation Forest and One-Class SVM are the same in the fair-budget and unified
runs, so those two are stable and reproduced. Only the autoencoder changed
between them, because the unified run standardizes its input.

## 2. What is verified

- **The score direction is correct.** Isolation Forest uses `-score_samples`,
  the autoencoder uses mean squared reconstruction error, and One-Class SVM uses
  `-decision_function`. In all three, a higher score means more novel (code in
  `classical_baselines.py` and `autoencoder_baseline.py`). One-Class SVM
  scores 0.96 with the same convention, so this is not a general sign mistake.
- **Below-chance AUROC means the detector rates SVHN as more normal than MNIST.**
  The mean scores in `SVHN_BASELINE_DIAGNOSTIC.md` show it: the autoencoder's
  mean error is 0.111 on MNIST and 0.020 on SVHN.
- **The raw autoencoder result is not seed noise.** Its standard deviation across
  seeds is 0.001. The standardized result is noisier (see section 3), and the
  three families of numbers (about 0.006, 0.760, 0.979) differ because of how the
  input was prepared and how much data was used.
- **SVHN embeddings are much shorter than MNIST ones.** The mean L2 length of the
  84-D embeddings is about 5.8 for SVHN (`svhn_inverted_embedding_norm_s42_v1`)
  and about 40.3 for MNIST validation (computed from the saved `val_features.npy`).
  Flipping SVHN pixel intensity changes the SVHN mean by only about 0.27, so the
  small size is not a polarity effect.

## 3. What the controlled check showed

Run `mnist_svhn_autoencoder_scaling_diagnostic_s42_46_v1` (new run, 2026-11-03).
Settings were fixed in `autoencoder_scaling_diagnostic.yaml` before the run:
seeds 42 to 46, network and forest settings copied from the earlier configs,
nothing tuned. MNIST validation vs SVHN test; the final MNIST test split was not
used. Because SVHN was seen in earlier runs, this is **exploratory**.

AUROC, mean ± sample SD over 5 seeds:

| Method | Input | 10,000 training rows | 49,000 training rows |
|---|---|---:|---:|
| Autoencoder | raw | 0.006 ± 0.001 | 0.010 ± 0.004 |
| Autoencoder | standardized | 0.979 ± 0.017 | 0.902 ± 0.072 |
| Isolation Forest | standardized | 0.280 ± 0.085 | 0.275 ± 0.043 |

Per-seed ranges: autoencoder standardized 0.951 to 0.993 (10,000 rows) and 0.803
to 0.978 (49,000 rows); Isolation Forest 0.14 to 0.36 and 0.23 to 0.33.

What this shows:

- **The new code reproduces the earlier runs.** At 10,000 rows the AUROC differs
  from the unified run by at most 2.2e-8 (autoencoder) and 0 (Isolation Forest),
  over the same five seeds.
- **Embedding length alone separates the two sets.** A control that scores each
  image by its embedding length gets AUROC 0.00007 (mean length 40.3 for MNIST
  validation, 5.8 for SVHN). So SVHN embeddings are far shorter, and ranking by
  "longer = more novel" is almost perfectly backwards.
- **The raw autoencoder behaves like that backwards length score** (0.006 and
  0.010) and does not change with training size. This fits the earlier hypothesis
  that its raw squared error mostly follows embedding size. It does not prove the
  mechanism.
- **Standardizing fixes the direction.** The same autoencoder scores 0.979 and
  0.902 once the input is standardized.
- **More training data did not help.** The 49,000-row result is lower (0.902 vs
  0.979) and less stable across seeds. No significance test was run, so treat the
  size of the gap as an observation only.
- **Isolation Forest is below chance in all 10 runs**, at both training sizes. It
  is stable, not a fluke, and standardizing did not change that.

## 4. What is still unexplained

- **The earlier 0.760 is not reproduced.** The earlier single run
  (`mnist_svhn_ae_train_standardized_s42_v1`) used seed 42, 49,000 rows, the same
  network and 20 epochs, with the scaler fitted on all 49,000 training embeddings.
  The new run gives 0.856 for seed 42 at 49,000 rows, and all five new 49,000-row
  seeds (0.803 to 0.978) are above 0.760. So seed choice alone does not explain
  it. The two code paths have not been compared line by line, and the cause is not
  identified.
- **Why Isolation Forest is below chance.** The idea that short SVHN embeddings sit
  nearer the middle of the standardized MNIST cloud than typical MNIST digits is
  still untested. The length control does not test it directly.
- **Why 49,000 rows scores lower than 10,000 rows** for the standardized
  autoencoder.

## 5. Decisions

- Scores are **not** reversed after seeing SVHN results, and no old run is edited.
  Reversing a score because it did badly on the evaluation set would be tuning on
  that set.
- In reports, show the standardized, same-budget, five-seed autoencoder result
  (0.979 ± 0.017) as the headline for the fair comparison, and show the
  49,000-row result (0.902 ± 0.072) next to it, because it uses the full training
  set and is lower and less stable. The raw result (about 0.006) is a separate
  "unscaled" row, with this note as the reason. Isolation Forest is reported as
  measured (0.280 ± 0.085) and described as "below chance on this embedding", not
  hidden.
- Do not call the 0.760 vs 0.902 vs 0.979 differences "explained". Section 4 lists
  what is open.
- Every number above is a development result. SVHN was examined before these
  explanations were written, so the analysis is post-hoc exploratory.

## Leakage control

This note uses only saved validation and SVHN scores and embedding statistics.
Nothing was fitted or tuned for it, and the official MNIST final-test set was
not used.
