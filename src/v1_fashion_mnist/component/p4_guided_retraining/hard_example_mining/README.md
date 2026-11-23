# `hard_example_mining` (Fashion-MNIST)

The code for this step is shared by both datasets and lives in [`src/v1_mnist/component/p4_guided_retraining/hard_example_mining/`](../../../../v1_mnist/component/p4_guided_retraining/hard_example_mining/).
Fashion-MNIST runs it with the settings in [`../../configs/`](../../configs/), so nothing is copied here.

The step produces the hard-image pools (SOM, random, max-softmax) and the curriculum comparison. Its results go to `outputs/v1_fashion_mnist/p4_guided_retraining/hard_example_mining/`.
