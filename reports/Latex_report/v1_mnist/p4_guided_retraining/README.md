# Report: p4_guided_retraining (`reports/Latex_report/v1_mnist/p4_guided_retraining/`)

Phase 4: hard example mining, the curriculum comparison, targeted augmentation and the SOM re-analysis. Figures in `fig/<step>/`; result tables (LaTeX, made from the result files) in `tables/`.

Source `guided_retraining.tex`, compiled `guided_retraining.pdf`, packages in `File_Setup.tex`, references in `references.bib`.

## How to build

The reports use `biblatex` with the `biber` back end. From this folder:

```bash
pdflatex guided_retraining
biber guided_retraining
pdflatex guided_retraining
pdflatex guided_retraining
```

(or `latexmk -pdf guided_retraining.tex`, or MiKTeX/TeXworks/Overleaf with biber selected). Build files (`.aux`, `.bbl`, `.bcf`, …) are ignored by git; the compiled PDF is kept.
