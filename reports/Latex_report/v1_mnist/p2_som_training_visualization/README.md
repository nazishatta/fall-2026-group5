# Report: p2_som_training_visualization (`reports/Latex_report/v1_mnist/p2_som_training_visualization/`)

Phase 2: SOM training and grid choice, cluster analysis and error patterns. Figures in `fig/som_nnsom_training/`, `fig/cluster_analysis/` and `fig/error_pattern_analysis/`.

Source `som_cluster_error.tex`, compiled `som_cluster_error.pdf`, packages in `File_Setup.tex`, references in `references.bib`.

## How to build

The reports use `biblatex` with the `biber` back end. From this folder:

```bash
pdflatex som_cluster_error
biber som_cluster_error
pdflatex som_cluster_error
pdflatex som_cluster_error
```

(or `latexmk -pdf som_cluster_error.tex`, or MiKTeX/TeXworks/Overleaf with biber selected). Build files (`.aux`, `.bbl`, `.bcf`, …) are ignored by git; the compiled PDF is kept.
