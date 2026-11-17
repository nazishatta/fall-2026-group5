# `held_out_class_novelty` (Fashion-MNIST)

The code for this step is shared by both datasets and lives in [`src/v1_mnist/component/p3_novelty_detection/held_out_class_novelty/`](../../../../v1_mnist/component/p3_novelty_detection/held_out_class_novelty/).
Fashion-MNIST runs it with the settings in [`../../configs/`](../../configs/), so nothing is copied here.

The step produces near novelty: networks trained on 6 classes, the other 4 treated as novel, over 5 class splits. Its results go to `outputs/v1_fashion_mnist/p3_novelty_detection/held_out_class_novelty/`.
