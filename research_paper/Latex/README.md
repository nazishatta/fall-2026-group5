# Paper: LaTeX source (`research_paper/Latex/`)

Journal paper for Neurocomputing (Elsevier), written with the `elsarticle` template (`review` option: double-spaced, as Elsevier asks for submissions; line numbers come from the `lineno` package).

- `neurocomputing_paper.tex`: the paper (front matter, highlights, abstract of at most 250 words, body, declarations).
- `references.bib`: the bibliography (copied from the report folder; the paper uses `elsarticle-num`).
- `Fig/`: the figures (PDF) and the two TikZ diagrams (`fig_pipeline.tex`, `fig_som.tex`); see `Fig/README.md`.
- `../scripts/label_error_maps.py`: adds hotspot labels and a key to the project's error maps (reads `outputs/`).

## Submission package

The journal builds the PDF from the uploaded LaTeX files, which it keeps in one folder, and asks for each figure as its own file named `Figure_1`, `Figure_2`, and so on. After building the paper (so the `.bbl` is current), run from the repository root:

```
python research_paper/scripts/package_submission.py highlights
python research_paper/scripts/package_submission.py package
```

This writes `research_paper/submission_package/` (git-ignored): `Figure_<n>.pdf` files in the order they appear (`a`/`b` for two-panel figures, `A1`/`A2` in the appendix), a copy of the `.tex` that uses those names, the `.bbl`, `.bib`, `Highlights.docx` and `cover_letter.tex`. `figure_list.tsv` maps each figure file to its source in `Fig/`. Upload every file in that folder.
