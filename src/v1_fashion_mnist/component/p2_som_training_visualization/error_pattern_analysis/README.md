# `error_pattern_analysis` (Fashion-MNIST)

The code for this step is shared by both datasets and lives in [`src/v1_mnist/component/p2_som_training_visualization/error_pattern_analysis/`](../../../../v1_mnist/component/p2_som_training_visualization/error_pattern_analysis/).
Fashion-MNIST runs it with the settings in [`../../configs/`](../../configs/), so nothing is copied here.

The step produces where the CNN's mistakes sit on the map: error hotspots, the held-out robustness check, and example images. Its results go to `outputs/v1_fashion_mnist/p2_som_training_visualization/error_pattern_analysis/`.
