# SOM grid documents (MNIST)

## What this folder is about

Our project trains a CNN to recognise handwritten digits (MNIST). We then use a **Self-Organizing Map (SOM)** to lay out the CNN's internal picture of each image on a 2D grid of cells (called "neurons"). Similar images land in nearby cells, so the map shows **where the CNN makes its mistakes**.

A SOM needs a grid size (10x10, 15x15, 20x20, ...). This folder answers two questions:

1. **Which grid size should we use, and was the choice fair?**
2. **Do our findings about the CNN's mistakes (RQ2) stay the same on other grid sizes?**

It also records honestly what went wrong earlier in the project and how we fixed it.

## Read the files in this order

| # | File | What it is |
|---|---|---|
| 1 | `SOM_GRID_SELECTION_PROTOCOL.md` | The original rules for choosing the grid size. Read the **"Recorded outcome"** section at the end first: it says what actually happened. |
| 2 | `DEVIATIONS.md` | Every time a rule was broken, and what we did about it. |
| 3 | `PREREGISTRATION_grid_comparison.md` | The rules for a new, fair test of 15x15 against 20x20, written **before** the test was run. |
| 4 | `prereg_stamp.txt` | A fingerprint (SHA-256) and timestamp of file 3. It proves the rules were written before any results existed. |
| 5 | `GRID_COMPARISON_RESULTS.md` | Results of that fair test. |
| 6 | `figures/grid_comparison_results.svg` | The chart for file 5. |
| 7 | `RQ2_ROBUSTNESS_CRITERIA.md` | The rules for checking whether our RQ2 findings also hold on 20x20 and 10x10, written **before** looking. These rules are meant to be reused for future datasets. |
| 8 | `RQ2_ROBUSTNESS_RESULTS.md` | Results of that check. |

## Why some files are never edited

Files 3, 4, 7 and the rules part of file 1 were written **before** the results. They are our evidence that we did not change the rules after seeing the results, so we leave them exactly as they were, even for wording. (Changing a single character in file 3 would no longer match its fingerprint in file 4.) Each one is explained in plain words in its results file.

## Words you will see

- **Grid size / neuron / cell:** a 15x15 map has 225 cells; each image is placed in the cell that matches it best.
- **QE (quantization error):** how far images are from their cell, on average. Lower means a closer fit.
- **TE (topological error):** how often similar images end up in cells that are not next to each other. Lower is better.
- **AUROC:** a score from 0.5 (guessing) to 1 (perfect) for how well something predicts the CNN's mistakes.
- **Pre-registration:** writing down the rules of a test before running it, so the results cannot steer the rules.
- **Headline model / robustness check:** the main model we report, and extra models used only to check that the results don't depend on that choice.

## Final result

- **Grid used: 15x15** (model `som_15x15_nb11_ep250_s42_v2`), as the instructor requested.
- **Fair test:** 15x15 and 20x20 were equally good at predicting where the CNN makes mistakes (AUROC 0.629 vs 0.630). With only 92 CNN mistakes the test cannot spot very small differences, but **nothing showed that a bigger grid is better**.
- **RQ2 holds on all three grids (15x15, 20x20, 10x10):** about 5% of the cells hold 10 to 13 times their fair share of the CNN's mistakes, and mistakes are most common in cells where different digits mix. Both findings are far beyond chance (p = 0.005).
- **In short: our main findings do not depend on the grid size we chose.**
