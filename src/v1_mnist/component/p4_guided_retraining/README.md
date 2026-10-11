# Phase 4: Guided Retraining

Uses the SOM's error hotspots to choose what the CNN is retrained on, then checks whether the map improves. No code yet.

```
p4_guided_retraining/
├── hard_example_mining/           1. proposal week 9: pick samples from hotspots and class boundaries
├── targeted_augmentation/         2. proposal week 10: augment only those samples
└── som_reanalysis/                3. proposal week 11: retrain, rebuild the SOM, compare before/after
```

**Order:** `hard_example_mining` → `targeted_augmentation` → `som_reanalysis`.
Compare against a random-selection control over at least 5 seeds.
When code is added: a runner in `pipeline/p4_guided_retraining/`, tests in `tests/p4_guided_retraining/`, results in `outputs/v1_mnist/p4_guided_retraining/`.
