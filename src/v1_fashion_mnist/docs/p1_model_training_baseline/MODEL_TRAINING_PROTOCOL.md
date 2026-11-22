# Protocol: model training baseline on Fashion-MNIST (Phase 1)

Part of the Fashion-MNIST protocol, one file per phase:
[Phase 1](MODEL_TRAINING_PROTOCOL.md) ·
[Phase 2](../p2_som_training_visualization/SOM_ANALYSIS_PROTOCOL.md) ·
[Phase 3](../p3_novelty_detection/NOVELTY_DETECTION_PROTOCOL.md) ·
[Phase 4](../p4_guided_retraining/RETRAINING_PROTOCOL.md).
Changes and deviations: [`../DEVIATIONS.md`](../DEVIATIONS.md). Why the study exists and the rule it follows:
[`../README.md`](../README.md).

Written on 2026-11-17, **before any Fashion-MNIST data was loaded, any network trained or any result seen**,
and committed as one file in `5dd164c` (`docs/FASHION_MNIST_PROTOCOL.md`). Split into one file per phase on
2026-11-21 with no rule changed (see `../DEVIATIONS.md`). Sections 1 to 4 are not changed; a change needs a
written reason in `../DEVIATIONS.md` and, for a run already made, a new run ID.

**Rule for every phase:** every step done for MNIST is done for Fashion-MNIST, with the MNIST design copied
unchanged, unless the dataset makes that impossible.

## 1. What is fixed before anything runs

| Item | Fixed value | Same as MNIST? |
|---|---|---|
| Data | Fashion-MNIST, 70,000 grayscale 28x28 images, 10 classes, `torchvision.datasets.FashionMNIST` (MIT licence; Xiao et al., 2017) | different dataset |
| Classes (torchvision order 0-9) | T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot | n/a |
| Split | the original 60,000 + 10,000 images joined (sample IDs 0-59,999 and 60,000-69,999), then a 70/15/15 stratified split with seed 42: 49,000 / 10,500 / 10,500 | yes |
| Preprocessing | pixel values divided by 255; nothing else | yes |
| Network | LeNet-5 and its full training recipe from `src/v1_mnist/component/configs/` (`base.yaml`: 20 epochs, Adam, learning rate 0.001, weight decay 0.0001, StepLR every 10 epochs by 0.1, batch 64, seed 42) | yes, **not tuned for Fashion-MNIST** |
| Embeddings | `fc2` (84 numbers) for every split, with sample IDs, labels, predictions and confidence | yes |
| Seed | CNN 42 | yes |

The network is deliberately not tuned. The aim is to see whether the pipeline, with MNIST's settings, still
finds structure on a harder dataset. Fashion-MNIST accuracy will therefore be lower than what a tuned network
reaches; that is expected and is not a weakness of the analysis.

## 2. Steps

- Train once with seed 42. Report accuracy with a 95% Wilson interval, macro-F1, per-class F1 and the
  confusion matrix.
- As for MNIST, the training script evaluates the test split **once**, at the end. Nothing (epochs,
  recipe, map, thresholds) is chosen from that number.
- No pass/fail.

## 3. Leakage control

- The Fashion-MNIST test split is used twice, both fixed in advance: once at the end of Phase 1 training
  (reported only), and once in the Phase 4 `final_test` stage. No setting is chosen from either.
- Every run records the SHA-256 of its inputs; a finished run is never overwritten.
- **Declared:** this protocol was written after the MNIST results were known. It copies MNIST's design
  instead of adapting it to those results, and no Fashion-MNIST result existed when it was written.

## 4. Limitations stated before the run

- The network is not tuned for Fashion-MNIST, so the results describe the MNIST recipe on Fashion-MNIST,
  not the best possible Fashion-MNIST setup.

## 5. Recorded outcome

Filled in after the run, without changing sections 1 to 4. Recorded 2026-11-21 from run
`lenet5_fashion_mnist_s42` (trained 2026-11-18, config SHA-256 `f6847755…`). Sources:
`outputs/v1_fashion_mnist/p1_model_training_baseline/cnn_baseline_training/metrics/lenet5_fashion_mnist_s42_run.json`,
`…_test_evaluation.json`, `logs/lenet5_fashion_mnist_s42_train_baseline.log`,
`embeddings/embedding_manifest.json` and `data_pipeline/metrics/split_stats.json`.

**Split.** 49,000 / 10,500 / 10,500 images, exactly 4,900 / 1,050 / 1,050 per class (10% of every split).

**Training.** Validation accuracy peaked at 91.66% in epoch 16, the checkpoint kept; training accuracy
93.24% at that checkpoint. The settings were used exactly as fixed in section 1; nothing was tuned.

**Test split, evaluated once at the end of training** (reported only; nothing was chosen from it):

| Measure | Fashion-MNIST | MNIST (for reference) |
|---|---|---|
| Accuracy | 90.90% (9,544 / 10,500) | 99.18% (10,414 / 10,500) |
| 95% Wilson interval | 90.33% to 91.43% | 98.99% to 99.34% |
| Macro-F1 | 0.909 | 0.992 |
| Test mistakes | 956 | 86 |
| Validation mistakes (best epoch) | 876 | 92 |

Per-class F1 on the test split: T-shirt/top 0.866, Trouser 0.987, Pullover 0.851, Dress 0.920, Coat 0.840,
Sandal 0.977, Shirt **0.746** (lowest), Sneaker 0.958, Bag 0.978, Ankle boot 0.968. The largest test
confusions are between upper-body garments: Shirt called T-shirt/top (110), T-shirt/top called Shirt (105),
Pullover called Coat (90), Shirt called Coat (79), Pullover and Coat called Shirt (62 each).

**Embeddings.** `fc2` (84 numbers) saved for all three splits with sample IDs, labels, predictions and
confidence; every value finite, every sample ID unique, counts matching the split; each file fingerprinted
with SHA-256 in the manifest. Mean confidence on validation: 0.942 when right, 0.686 when wrong
(MNIST: 0.997 and 0.833).

**Reading.** As expected in section 1, the untuned MNIST recipe reaches clearly lower accuracy on
Fashion-MNIST, and its mistakes concentrate in a few similar classes. With about ten times as many
validation mistakes as MNIST, Phases 2 to 4 have much more to study. No deviation from sections 1 to 4.
Report: `reports/Latex_report/v1_fashion_mnist/p1_model_training_baseline/cnn_baseline_report.tex`.

## 6. References

Keys in `reports/Latex_report/v1_mnist/Final_report/references.bib`.

- Xiao, H., Rasul, K. and Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine
  learning algorithms. arXiv:1708.07747. (`xiao2017fashion`)
