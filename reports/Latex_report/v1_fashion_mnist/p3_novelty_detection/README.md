# p3_novelty_detection report (Fashion-MNIST)

LaTeX report for p3_novelty_detection on Fashion-MNIST. Phase 3, novelty detection: can the map flag images unlike Fashion-MNIST?

Source `novelty_detection.tex`, packages in `File_Setup.tex`, references in `references.bib`. Figures go in `fig/qe_novelty_detection/`, `fig/u_matrix_analysis/` and `fig/novelty_benchmark/`, copied from the `pdf/` folders of the runs in `outputs/v1_fashion_mnist/p3_novelty_detection/` (kept in DVC). Until a figure is copied, the report shows a box marked [TBD] in its place.

Build from this folder: `latexmk -pdf novelty_detection.tex` (biber is run automatically). Build files are ignored by git; the compiled PDF is kept.
