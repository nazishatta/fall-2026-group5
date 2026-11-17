# Final report (`reports/Latex_report/v1_mnist/Final_report/`)

The whole MNIST study in one report: the CNN baseline, the SOM and its error patterns, novelty detection (SVHN and held-out digits), guided retraining and the SOM re-analysis, with a leakage-control section and the limitations.

| Path | What it is |
|---|---|
| `final_report.tex` / `final_report.pdf` | Source and compiled report |
| `File_Setup.tex` | Packages and page style |
| `references.bib` | Every cited work |
| `fig/` | The figures (PDF, kept in git so the report builds from a clean clone). Names start with the phase (`p1_…` to `p4_…`). |
| `fig/diagrams/` | The four draw.io diagrams (project pipeline, how a SOM works, the retraining loop, the end-to-end comparison), each as `.drawio`, `.drawio.svg` and `.pdf` |
| `tables/` | Tables kept in their own `.tex` files |

In the retraining table, the new-SOM values of the least-confident arm (marked †) come from a secondary run added after the other results were seen (`MSP_ARM_ADDENDUM.md`); the curriculum arm was not re-analysed (n/a).

## How to build

The reports use `biblatex` with the `biber` back end. From this folder:

```bash
pdflatex final_report
biber final_report
pdflatex final_report
pdflatex final_report
```

(or `latexmk -pdf final_report.tex`, or MiKTeX/TeXworks/Overleaf with biber selected). Build files (`.aux`, `.bbl`, `.bcf`, …) are ignored by git; the compiled PDF is kept.
