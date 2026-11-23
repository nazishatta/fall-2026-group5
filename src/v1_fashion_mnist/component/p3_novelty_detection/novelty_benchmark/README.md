# `novelty_benchmark` (Fashion-MNIST)

The code for this step is shared by both datasets and lives in [`src/v1_mnist/component/p3_novelty_detection/novelty_benchmark/`](../../../../v1_mnist/component/p3_novelty_detection/novelty_benchmark/).
Fashion-MNIST runs it with the settings in [`../../configs/`](../../configs/), so nothing is copied here.

The step produces the SOM scores against max-softmax, energy, Mahalanobis and nearest-neighbour baselines on the same embeddings. Its results go to `outputs/v1_fashion_mnist/p3_novelty_detection/novelty_benchmark/`.
