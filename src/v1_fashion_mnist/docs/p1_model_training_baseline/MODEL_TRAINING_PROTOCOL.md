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

Filled in after the run, without changing sections 1 to 4. [TBD]

## 6. References

Keys in `reports/Latex_report/v1_mnist/Final_report/references.bib`.

- Xiao, H., Rasul, K. and Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine
  learning algorithms. arXiv:1708.07747. (`xiao2017fashion`)
