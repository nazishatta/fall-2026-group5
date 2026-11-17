# Report: p1_model_training_baseline (`reports/Latex_report/v1_mnist/p1_model_training_baseline/`)

Phase 1: the data split, the LeNet-5 baseline, its test results (99.18% accuracy) and the fc2 embeddings. Figures in `fig/cnn_baseline/`.

Source `cnn_baseline_report.tex`, compiled `cnn_baseline_report.pdf`, packages in `File_Setup.tex`, references in `references.bib`.

## How to build

The reports use `biblatex` with the `biber` back end. From this folder:

```bash
pdflatex cnn_baseline_report
biber cnn_baseline_report
pdflatex cnn_baseline_report
pdflatex cnn_baseline_report
```

(or `latexmk -pdf cnn_baseline_report.tex`, or MiKTeX/TeXworks/Overleaf with biber selected). Build files (`.aux`, `.bbl`, `.bcf`, …) are ignored by git; the compiled PDF is kept.
