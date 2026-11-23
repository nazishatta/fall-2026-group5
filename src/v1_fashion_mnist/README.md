# Fashion-MNIST study (`v1_fashion_mnist`)

The same pipeline as the MNIST study, run on Fashion-MNIST (70,000 grayscale 28x28 images of 10 clothing classes).

| Folder | What it holds |
|---|---|
| [`component/configs/`](component/configs/README.md) | settings for every step; `base.yaml` sends every result to `outputs/v1_fashion_mnist/` |
| [`component/pipeline/`](component/pipeline/README.md) | the runner, one subfolder per phase |
| `component/p1_...` to `component/p4_...` | one folder per step, pointing to the shared code in `src/v1_mnist/component/` |
| [`docs/`](docs/README.md) | one protocol per phase (written before any run), `DEVIATIONS.md` and results notes |
| [`tests/`](tests/README.md) | tests for the Fashion-MNIST settings and runner |

The step code is shared with MNIST, so a fix there applies to both datasets. Protocols: one per phase, listed in [`docs/README.md`](docs/README.md); changes in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).
