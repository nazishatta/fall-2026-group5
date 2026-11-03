# Phase 4: Guided Retraining

Uses the SOM's error hotspots to choose what the CNN is retrained on, then checks whether the map improves. `hard_example_mining` decides which training images are hard and compares standard training with curricula; `targeted_augmentation` adds altered copies of those images and retrains. `som_reanalysis` has no code yet.

```
p4_guided_retraining/
├── hard_example_mining/           1. pools of hard images, curriculum comparison (see its README)
├── targeted_augmentation/         2. augment only those images, retrain (see its README)
└── som_reanalysis/                3. retrain, rebuild the SOM, compare before/after
```

**Order:** `hard_example_mining` → `targeted_augmentation` → `som_reanalysis`.
Compare against a random-selection control over at least 5 seeds.
Runners are in `pipeline/p4_guided_retraining/`, tests in `tests/p4_guided_retraining/`, results in `outputs/v1_mnist/p4_guided_retraining/`. Code used by both steps (frozen split, CNN trainer, metrics, paired statistics, protocol stamp) is in `utils/`.

**Protocol:** `docs/p4_guided_retraining/RETRAINING_PROTOCOL.md`, frozen before any test use.

**Running it:** see [`HOW_TO_RUN.md`](HOW_TO_RUN.md) (EC2 commands, from environment set-up to the pull request).
