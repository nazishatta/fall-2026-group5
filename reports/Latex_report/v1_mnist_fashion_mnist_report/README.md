# Combined capstone report: MNIST and Fashion-MNIST (`reports/Latex_report/v1_mnist_fashion_mnist_report/`)

The formal capstone report for the whole project, on the theme of the project proposal
(`proposals/Project_proposal.md`): *Post-Training Analysis and Novelty Detection Using Neural Network
Self-Organizing Maps (NNSOM)*. It merges the results of both datasets, side by side, from the two final reports:
`../v1_mnist/Final_report/` and `../v1_fashion_mnist/Final_report/`. Every number is copied from those reports
(and through them from the phase reports and their result files).

The layout follows the program's sample capstone report: a title page with the abstract, a table of contents,
and the sections Introduction, Problem Statement, Related Work, Solution and Methodology, Results and Discussion
(Experimentation protocol, Data tables, Graphs), Discussion, Conclusion and References.

| Path | What it is |
|---|---|
| `combined_report_v2.tex` | Source of the report (the earlier `combined_report.tex` was removed on 2026-11-26; it is in git history) |
| `File_Setup.tex` | Page and package setup from the sample report |
| `references.bib` | Every cited work (BibTeX, IEEEtran style) |
| `fig/` | PDF figures: the GW logo, and figures copied from both final reports, prefixed `mnist_` or `fashion_mnist_` |

## How to build

This report uses BibTeX (as the sample does), not biber. From this folder:

```bash
latexmk -pdf combined_report_v2.tex
```

or `pdflatex combined_report_v2`, `bibtex combined_report_v2`, then `pdflatex combined_report_v2` twice. Build files
(`.aux`, `.bbl`, `.blg`, …) are ignored by git.
