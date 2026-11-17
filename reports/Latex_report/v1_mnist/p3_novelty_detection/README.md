# Report: p3_novelty_detection (`reports/Latex_report/v1_mnist/p3_novelty_detection/`)

Phase 3: QE and U-Matrix novelty detection on MNIST against SVHN, and the comparison with the baseline detectors. Figures in `fig/qe_novelty_detection/`, `fig/u_matrix_analysis/`, `fig/novelty_benchmark/`, and the draw.io diagram in `fig/novelty_framework/`.

Source `novelty_detection.tex`, compiled `novelty_detection.pdf`, packages in `File_Setup.tex`, references in `references.bib`.

## How to build

The reports use `biblatex` with the `biber` back end. From this folder:

```bash
pdflatex novelty_detection
biber novelty_detection
pdflatex novelty_detection
pdflatex novelty_detection
```

(or `latexmk -pdf novelty_detection.tex`, or MiKTeX/TeXworks/Overleaf with biber selected). Build files (`.aux`, `.bbl`, `.bcf`, …) are ignored by git; the compiled PDF is kept.
