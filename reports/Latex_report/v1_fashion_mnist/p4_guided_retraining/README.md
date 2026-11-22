# p4_guided_retraining report (Fashion-MNIST)

LaTeX report for Phase 4 on Fashion-MNIST, guided retraining: does retraining on images the SOM picks beat
picking at random? It follows the MNIST Phase 4 report (`../../v1_mnist/p4_guided_retraining/`) section by
section, with a Fashion-MNIST/MNIST comparison table.

## Files

| File | What it is |
|---|---|
| `guided_retraining.tex` | The report (written by hand; every number copied from the result files below) |
| `File_Setup.tex` | Shared preamble (same as the other reports) |
| `references.bib` | Bibliography (the MNIST Phase 4 list plus SVHN) |
| `fig/` | PDF figures copied from `outputs/v1_fashion_mnist/p4_guided_retraining/` (see `fig/README.md`) |

## Sources

All under `outputs/v1_fashion_mnist/p4_guided_retraining/`:

- `hard_example_mining/hard_example_pools_som_15x15_nb11_ep250_s42/` (pools)
- `targeted_augmentation/augmentation_comparison_som_15x15_nb11_ep250_s42_r3/` (augmentation arms, revision r3; the
  same folder without a suffix (r1) and with `_r2` hold the two failed transform checks)
- `hard_example_mining/curriculum_comparison_som_15x15_nb11_ep250_s42/` (curriculum arms)
- `som_reanalysis/som_reanalysis_som_15x15_nb11_ep250_s42/` and `..._msp_arm_...` (new SOMs)
- `novelty_after_retraining/novelty_after_retraining_som_15x15_nb11_ep250_s42/` (novelty after retraining)

Protocol and recorded outcome: `src/v1_fashion_mnist/docs/p4_guided_retraining/RETRAINING_PROTOCOL.md` and
`RETRAINING_RESULTS.md`; deviations in `src/v1_fashion_mnist/docs/DEVIATIONS.md`.

## Build

```
latexmk -pdf guided_retraining.tex
```
