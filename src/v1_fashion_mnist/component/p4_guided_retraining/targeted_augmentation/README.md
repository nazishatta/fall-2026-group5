# `targeted_augmentation` (Fashion-MNIST)

The code for this step is shared by both datasets and lives in [`src/v1_mnist/component/p4_guided_retraining/targeted_augmentation/`](../../../../v1_mnist/component/p4_guided_retraining/targeted_augmentation/).
Fashion-MNIST runs it with the settings in [`../../configs/`](../../configs/), so nothing is copied here.

The step produces retraining with altered copies of the pool images, 5 seeds per arm. Its results go to `outputs/v1_fashion_mnist/p4_guided_retraining/targeted_augmentation/`.
