# Source code (`src/`)

One folder per dataset, each with the same phase folders:

- [`v1_mnist/`](v1_mnist/README.md): the MNIST study. Its `component/` folder holds the step code, which is shared by both datasets.
- [`v1_fashion_mnist/`](v1_fashion_mnist/README.md): the Fashion-MNIST study. It holds only what differs: its settings (`component/configs/`), its runner (`component/pipeline/`), its protocol and notes (`docs/`) and its tests. Every step runs the shared code in `v1_mnist/component/` with these settings, and the results go to `outputs/v1_fashion_mnist/`.
