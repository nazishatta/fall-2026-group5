# `data_pipeline` (Fashion-MNIST)

The code for this step is shared by both datasets and lives in [`src/v1_mnist/component/p1_model_training_baseline/data_pipeline/`](../../../../v1_mnist/component/p1_model_training_baseline/data_pipeline/).
Fashion-MNIST runs it with the settings in [`../../configs/`](../../configs/), so nothing is copied here.

The step produces the 70/15/15 stratified split of the 70,000 Fashion-MNIST images (seed 42) and the class counts per split. Its results go to `outputs/v1_fashion_mnist/p1_model_training_baseline/data_pipeline/`.
