# `som_nnsom_training` (Fashion-MNIST)

The code for this step is shared by both datasets and lives in [`src/v1_mnist/component/p2_som_training_visualization/som_nnsom_training/`](../../../../v1_mnist/component/p2_som_training_visualization/som_nnsom_training/).
Fashion-MNIST runs it with the settings in [`../../configs/`](../../configs/), so nothing is copied here.

The step produces the 15x15 NNSOM map (neighbourhood 11, 250 epochs, seed 42) trained on the training embeddings, its quality measures and figures. Its results go to `outputs/v1_fashion_mnist/p2_som_training_visualization/som_nnsom_training/`.
