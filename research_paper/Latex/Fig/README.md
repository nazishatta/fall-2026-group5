# Paper figures (`research_paper/Latex/Fig/`)

Files used by `neurocomputing_paper.tex`:

- `diagram_pipeline.pdf`, `diagram_som.pdf` and `diagram_cf1_cf2.pdf`: the diagrams (pipeline; how the SOM works; CF1 against CF2). Their TikZ sources are in `research_paper/scripts/diagrams/` (build each with `pdflatex pipeline.tex`, `som.tex`, `endtoend.tex` there and copy the PDF here).
- `error_map_mnist_labeled.pdf` and `error_map_fashion_mnist_labeled.pdf`: the project's own validation error maps (`.../error_pattern_analysis/figures/<run>/pdf/analysis_maps/validation_complex_error_map.pdf`), unchanged, on a larger page with hotspot outlines, hotspot labels (true class -> predicted class) and a key. Made by `research_paper/scripts/label_error_maps.py`.
- `mnist_p4_confusion_standard_vs_augmented_som.pdf` and `fashion_mnist_p4_confusion_standard_vs_augmented_som.pdf`: the project's own CF1 / CF2 confusion figures (unchanged).
- `svhn_som_overlay_mnist.pdf` and `svhn_som_overlay_fashion_mnist.pdf`: SVHN images on the frozen SOMs (the report's own figures; from `outputs/v1_fashion_mnist/p3_novelty_detection/u_matrix_analysis/fashion_mnist_svhn_som_overlay_s42_v1/pdf/` and `reports/Latex_report/v1_mnist_fashion_mnist_report/fig/mnist_p3_svhn_som_overlay.pdf`).
- `mnist_p2_u_matrix.pdf` and `fashion_mnist_p2_u_matrix.pdf`, `mnist_p4_map_measures_by_arm.pdf` and `fashion_mnist_p4_map_measures_by_arm.pdf`: U-matrices and measures of the new SOMs after retraining (the report's own figures, copied from `reports/Latex_report/v1_mnist_fashion_mnist_report/fig/`).
- `hotspot_examples_mnist.pdf` and `hotspot_examples_fashion_mnist.pdf` (Appendix A): one representative error per hotspot cell (copied from `outputs/.../representative_samples/pdf/hotspot_representatives.pdf` and cropped to the figure area).

Figure titles: the guide for authors asks that a figure's title appear only in its caption. The title line printed at the top of the U-matrix, SVHN overlay, map-measure and hotspot-example figures was therefore removed by moving the page box (`research_paper/scripts/package_submission.py strip-titles`); nothing inside the plots changed. The originals in `outputs/` and `reports/` are untouched.
