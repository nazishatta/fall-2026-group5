# Post-Training Analysis and Novelty Detection Using Neural Network Self-Organizing Maps

**Do neural network errors have geography?** This project tests whether the topology of a
trained model's learned representation — mapped with a Self-Organizing Map — can locate
where a model systematically fails, recognize inputs it has never seen, and guide targeted
intervention that measurably improves it.

> **Status:** in development. Sections marked `[TBD]` are placeholders and will be filled
> with real measured values. No result is reported here before it exists.

---

## Why this matters

A model can post strong average accuracy while hiding structured weaknesses. Existing
interpretability tools — saliency maps, SHAP, LIME — explain one prediction at a time.
They answer *which pixels mattered for this sample*, not *where in the representation do
this model's failures concentrate*. That population-level view is what determines whether
a model is safe to deploy, and it is what this project tries to recover.

## Central research hypothesis

> SOM-based topology and quantization signals can expose structured regions of model
> failure in learned representation space; those signals can detect distributional
> novelty; and SOM-derived diagnostics can guide targeted interventions that measurably
> improve predictive behavior and/or the structure of the learned representation.

This is treated as a hypothesis to test, not a conclusion to prove. Components
unsupported by evidence will be reported as unsupported.

## Pipeline

```
Dataset → Baseline CNN → Baseline Evaluation → Feature Extraction → PCA (optional)
   → SOM Training → Representation Analysis → Error Geography
   → Novelty / OOD Detection → Baseline Comparison
   → SOM-Guided Intervention → Retraining → Feature Re-extraction
   → SOM Re-analysis → Before/After Comparison
```

## Research questions

| | Question | Primary output |
|---|---|---|
| **RQ1** | What does the latent feature space look like under a topology-preserving projection? | Cluster structure, class overlap, density maps |
| **RQ2** | Do misclassifications occupy identifiable regions of the representation? | Error hotspot statistics + representative samples |
| **RQ3** | Can SOM-derived measures identify out-of-distribution observations? | AUROC / AUPR / FPR@95TPR |
| **RQ4** | How does SOM novelty detection compare with established baselines? | Isolation Forest, One-Class SVM, autoencoder |
| **RQ5** | Can SOM-derived error regions identify samples worth targeting in retraining? | Guided vs. **random-selection control** |
| **RQ6** | After intervention, is the representation measurably better structured? | Before/after representation metrics |

Success criteria for each RQ are fixed **before** the test set is examined
(`docs/success_criteria.md`).

## Key results

`[TBD — populated from results/tables/ once experiments run]`

| Metric | Baseline | SOM-guided | Random control |
|---|---|---|---|
| Macro-F1 | `[TBD]` | `[TBD]` | `[TBD]` |
| Hotspot error rate | `[TBD]` | `[TBD]` | `[TBD]` |
| OOD AUROC | `[TBD]` | `[TBD]` | — |

## Repository structure

```
configs/      experiment configuration (YAML) — no settings hidden in notebooks
data/         raw + processed data (gitignored; regenerated from code)
src/          all reusable logic
  data/       dataset registry and splitting
  models/     architectures and training
  features/   embedding extraction with provenance metadata
  som/        NNSOM training and diagnostics
  analysis/   representation analysis, error geography
  ood/        novelty detection and baselines
  retraining/ SOM-guided interventions and controls
  evaluation/ metrics and evaluation protocols
  visualization/ reusable plotting
  utils/      config, seeding, paths
scripts/      entry points for each pipeline stage
notebooks/    exploration and narrative only — not a home for logic
tests/        tests for reusable logic
results/      figures, tables, metrics, per-experiment outputs
reports/      paper and presentation
```

## Installation

```bash
git clone https://github.com/<org-or-user>/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
# Run the test suite
pytest -q

# Train the baseline (once scripts/train_baseline.py is implemented)
python scripts/train_baseline.py --config configs/base.yaml

# Switch datasets without touching code
python scripts/train_baseline.py --config configs/base.yaml \
    --override configs/dataset/fashion_mnist.yaml
```

## Reproduction

Every reported number traces back through
**config → raw result → processed metric → table/figure → conclusion**.
Experiment names encode their configuration, e.g.
`fashion_mnist_lenet5_penultimate_som15x15_seed42`.

## Data

MNIST, Fashion-MNIST and CIFAR-10 download automatically via torchvision on first run.
See `data/README.md` for the embedding metadata schema.

## Technologies

PyTorch · [NNSOM](https://amir-jafari.github.io/SOM/) · scikit-learn · NumPy · Matplotlib · MLflow · pytest

## Limitations

`[TBD — written from actual findings, not anticipated ones]`

## Team

- `[Name]` — `[role]`
- `[Name]` — `[role]`

Advisor: Dr. Amir Jafari, The George Washington University, Data Science Program.
Developed as a capstone project in the GWU MS Data Science program.

## License

See [LICENSE](LICENSE).
