# Integrated MNIST CNN–NNSOM manuscript — Prism-ready LaTeX bundle

Import this ZIP directly into Prism as a LaTeX project. Set `main.tex` as the root document if necessary.

Contents:
- `main.tex`: single integrated research-paper manuscript (original research text and results preserved).
- `research_paper.pdf`: compiled PDF from `main.tex`.
- `File_Setup.tex`, `references.bib`, `tables/`: required dependencies.
- `fig/*.pdf`: original experiment figures, unchanged.
- `fig/pipeline/pipeline.drawio`: editable diagram.
- `fig/pipeline/pipeline.drawio.svg`: vector source export.
- `fig/pipeline/pipeline.pdf`: vector figure embedded in `main.tex` as Figure 1.

Build (from project root): `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.

Only the pipeline figure environment in the original `final_report.tex` was replaced, and the main manuscript renamed `main.tex`. No original research data, tables, figures, or results were altered. The PDF is a working draft, not a journal submission confirmation.
