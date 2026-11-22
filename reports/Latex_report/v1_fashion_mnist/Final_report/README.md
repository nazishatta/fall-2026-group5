# Final report (`reports/Latex_report/v1_fashion_mnist/Final_report/`)

The whole Fashion-MNIST study in one report, with the same sections as the MNIST final report
(`../../v1_mnist/Final_report/`): the CNN baseline, the SOM and its error patterns, novelty detection (SVHN and
unseen clothing classes), guided retraining with the SOM re-analysis and novelty after retraining, an end-to-end
comparison, a Fashion-MNIST/MNIST comparison table, leakage control and limitations. Every number is copied from
the four Fashion-MNIST phase reports in `../p1_model_training_baseline/` to `../p4_guided_retraining/`, which name
their result files.

| Path | What it is |
|---|---|
| `final_report.tex` | Source of the report |
| `File_Setup.tex` | Packages and page style (same as MNIST) |
| `references.bib` | Every cited work (the MNIST final report's list) |
| `fig/` | The figures (PDF), copied from the phase reports. Names start with the phase (`p2_…` to `p4_…`). |
| `tables/` | Tables kept in their own `.tex` files |

In the retraining table, the new-SOM values of the least-confident arm (marked †) come from the secondary
re-analysis run, as on MNIST; n/a marks values that were not part of a run.

## How to build

The reports use `biblatex` with the `biber` back end. From this folder:

```bash
latexmk -pdf final_report.tex
```

(or `pdflatex final_report`, `biber final_report`, then `pdflatex final_report` twice). Build files (`.aux`,
`.bbl`, `.bcf`, …) are ignored by git.
