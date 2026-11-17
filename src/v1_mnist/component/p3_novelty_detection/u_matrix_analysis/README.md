# U-Matrix analysis (`src/v1_mnist/component/p3_novelty_detection/u_matrix_analysis/`)

Proposal step: U-Matrix & Topological Analysis. The U-Matrix shows how far each neuron is from its neighbours, so it marks borders between regions of the map. This step scores novelty with the U-Matrix, compares it with QE, draws where the SVHN images land on the map (and where QE misses them), and checks the work against the proposal.

Runner: `pipeline/p3_novelty_detection/run_u_matrix_analysis.py`. Results: `outputs/v1_mnist/p3_novelty_detection/u_matrix_analysis/`.

## Files

| File | What it is |
|---|---|
| `novelty_overlay.py` | Phase 3 validation-only SOM overlay and descriptive miss analysis. |
| `run_proposal_gap_analysis.py` ▶ | Run a new Phase 3 *descriptive* analysis without modifying historical outputs. |
| `run_qe_u_matrix.py` ▶ | Reproduce MNIST validation U-Matrix statistics using the frozen SOM. |
| `run_qe_umatrix_validation_figure.py` ▶ | Reproduce the MNIST validation QE versus U-Matrix visualization. |
| `run_svhn_som_overlay.py` ▶ | Save SVHN-to-SOM assignments and describe where missed novel images land. |
| `run_u_matrix_grid_figure.py` ▶ | Plot the frozen SOM's U-Matrix on its actual hexagonal topology. |
| `svhn_som_overlay.py` | SVHN on the SOM grid: where novel images land and how ordinary they look to the SOM. |
| `u_matrix_novelty.py` | U-Matrix scores using the existing SOM neighbor rule. |

▶ = a script that can also be run on its own. The runners in `component/pipeline/` call these scripts in order.
