# CNN–NNSOM Integrated Research Paper

**Research Title:** Where Do Neural Networks Fail? Post-Training Error Geography, Novelty Detection, and Guided Retraining with Neural Network Self-Organizing Maps

## Research Overview

This research investigates Neural Network Self-Organizing Maps (NNSOM) as post-training tools for interpreting CNN classification errors, detecting unfamiliar inputs, and guiding targeted retraining.

The study uses a LeNet-5 convolutional neural network and the NNSOM library to analyze learned feature representations, identify class-confusion regions, evaluate novelty detection, and investigate whether SOM-guided interventions improve classification performance.

The primary contribution is an interpretable, topology-preserving representation of CNN error regions, evaluated against conventional baseline methods.

## Completed Research — MNIST

| Phase | Research Work | Status |
|---|---|---|
| Phase 1 | LeNet-5 CNN baseline, classification evaluation, and feature extraction | Completed |
| Phase 2 | NNSOM training, visualization, class-cluster analysis, and error hotspots | Completed |
| Phase 3 | MNIST-versus-SVHN novelty detection and conventional baseline comparisons | Completed |
| Phase 4 | Hard-example mining, targeted augmentation, curriculum learning, and SOM re-analysis | Completed |
| Research Paper | Integrated MNIST methodology, experimental results, figures, and discussion | Draft complete |

*Completed refers to the research evidence incorporated into the manuscript; it does not imply that all repository reviews or publication requirements are finished.*

## Current Research Findings

The manuscript reports that SOM-based analysis identifies interpretable regions of class confusion and concentrated CNN errors.

For novelty detection, conventional methods such as maximum-softmax probability and energy scoring outperform SOM quantization error on the evaluated MNIST-versus-SVHN benchmark.

SOM-guided targeted augmentation improves the evaluated hotspot error rate relative to standard and random training controls, although confidence-guided augmentation remains competitive.

These results support the interpretability and diagnostic value of NNSOM without claiming superiority across all performance metrics.

## Planned Work — Fashion-MNIST and Paper Finalization

The following work remains pending:

- Apply the same LeNet-5 CNN architecture and corresponding NNSOM analysis pipeline to Fashion-MNIST.
- Incorporate verified Fashion-MNIST classification, SOM analysis, novelty detection, and retraining results into the existing manuscript.
- Compare findings across MNIST and Fashion-MNIST using consistent evaluation procedures.
- Complete final scientific verification and update any outstanding experimental limitations.
- Select an appropriate peer-reviewed journal and adapt the manuscript to its official LaTeX template.
- Complete team review, repository integration, and journal-submission preparation.

Fashion-MNIST results will be added only after the corresponding experiments have been completed and verified.

## Manuscript Files

| File | Description |
|---|---|
| `main.tex` | Editable integrated LaTeX manuscript |
| `research_paper.pdf` | Compiled research paper |
| `File_Setup.tex` | LaTeX formatting and configuration |
| `references.bib` | Scientific bibliography |
| `fig/` | Experimental PDF figures |
| `fig/pipeline/` | Editable draw.io diagram and vector exports |
| `tables/` | Supporting research result tables |

## Compilation

From the `research_paper/Latex/` directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

**Manuscript status:** Integrated MNIST research draft complete. Fashion-MNIST extension, final scientific review, and journal submission pending.
