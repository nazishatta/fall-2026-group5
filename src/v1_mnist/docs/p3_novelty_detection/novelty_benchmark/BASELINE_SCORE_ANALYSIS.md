# Why Isolation Forest and the autoencoder look odd on SVHN

Recorded: 2026-11-03. Development results only (MNIST validation vs SVHN test).
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
- **The autoencoder result is not seed noise.** Its standard deviation across
  seeds is 0.001 (raw) and 0.017 (standardized). The three families of numbers
  (about 0.006, 0.760, 0.979) differ because of how the input was prepared and
  how much data was used.
- **SVHN embeddings are much shorter than MNIST ones.** The mean L2 length of the
  84-D embeddings is about 5.8 for SVHN (`svhn_inverted_embedding_norm_s42_v1`)
  and about 40.3 for MNIST validation (computed from the saved `val_features.npy`).
  Flipping SVHN pixel intensity changes the SVHN mean by only about 0.27, so the
  small size is not a polarity effect.

## 3. What is only a hypothesis

These explanations fit the numbers but have **not** been tested.

- **Raw autoencoder.** The error is an absolute squared difference, so
  small-magnitude inputs are reconstructed with a small error whatever they are.
  SVHN embeddings are about 7 times shorter than MNIST ones, so they would get
  low errors and look "normal". This would explain a near-zero AUROC.
- **Isolation Forest.** Points near the middle of the training cloud need many
  splits to isolate, so they look normal. Short SVHN embeddings may sit nearer the
  middle of the scaled MNIST cloud than typical MNIST digits do.
- **Standardized autoencoder, 0.760 vs 0.979.** Both runs use the same network
  and 20 epochs. They differ in training size (49,000 vs 10,000, so about five
  times more training steps) and in which data the scaler was fitted on. Which of
  these matters, if either, is **not explained**.

## 4. Decisions

- Scores are **not** reversed after seeing SVHN results, and no old run is edited.
  Reversing a score because it did badly on the evaluation set would be tuning on
  that set.
- In reports, the autoencoder's headline number is the standardized, same-budget,
  five-seed result (0.979 ± 0.017). The raw result (about 0.006) is shown as a
  separate "unscaled" row, with this note as the reason for the difference.
  Isolation Forest is reported as measured (0.280 ± 0.085) and described as
  "below chance on this embedding", not hidden.
- Every number above is a development result. SVHN was examined before these
  explanations were written, so the analysis is post-hoc exploratory.

## 5. Planned check (not run yet)

A new run under a new run ID, on validation data only, with the settings fixed
before it is run:

- Vary one thing at a time: training size (10,000 vs 49,000) and input scaling
  (raw vs standardized), with the same five seeds and the same network.
- Add a length-only control score (the negative embedding length) to show how
  much of each result comes from embedding size alone.

Results: [TBD] until the run exists.

## Leakage control

This note uses only saved validation and SVHN scores and embedding statistics.
Nothing was fitted or tuned for it, and the official MNIST final-test set was
not used.
