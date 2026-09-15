# Week 3 Final Report

## Post-Training Analysis Using NNSOM

### Status

**Week 3 scientific work is complete and frozen.**

Selected development SOM: **20x20, seed 42**

NNSOM version: **1.8.3**

Selected model SHA-256:

`528bb5e11cac077c155ba42855ed1c423aba7d0e70d86f5f42a0c7dee813c576`

Final test evaluated during Week 3 development: **No**

---

## 1. Objective

Week 3 established a reproducible Self-Organizing Map analysis layer
over the frozen Week 2 LeNet-5 latent embeddings.

Completed work:

- NNSOM API validation
- reproducible SOM training
- memory-safe initialization
- grid-size comparison
- frozen grid-selection protocol
- train/validation quality evaluation
- BMU assignment
- RQ1 representation analysis
- RQ2 error-geography analysis
- machine-readable outputs
- publication-oriented figures
- regression and integrity checks

---

## 2. Frozen Experimental Setting

- Dataset: MNIST
- Feature layer: LeNet-5 fc2
- Feature dimension: 84
- Training embeddings: 49,000
- Validation embeddings: 10,500
- Development seed: 42
- Selected grid: 20x20

---

## 3. Grid Selection

| Grid | Validation QE | TE1 (%) | TE1+2 (%) | Occupancy | Rank Sum |
|---|---:|---:|---:|---:|---:|
| 10x10 | 1.342208 | 7.4381 | 1.1905 | 0.9900 | 6 |
| 15x15 | 1.132639 | 9.2476 | 1.6095 | 0.8889 | 8 |
| **20x20** | **0.968939** | **5.5333** | 1.3810 | 0.8150 | **4** |

The prospectively frozen equal-weight ordinal rank-sum rule selected
20x20.

The selected grid does not dominate every metric: 10x10 has the best
TE1+2 and greater occupancy. This disagreement is retained as a
reported trade-off.

---

## 4. RQ1 — Representation Structure

### Training

- Occupancy: 0.8850
- Occupied neurons: 354/400
- Sample-weighted purity: 0.985918
- Normalized entropy: 0.019193

### Validation

- Occupancy: 0.8150
- Occupied neurons: 326/400
- Sample-weighted purity: 0.978762
- Normalized entropy: 0.027479

### Interpretation

The SOM shows strongly class-organized latent structure, while mixed
neurons remain and provide useful regions for error analysis.

---

## 5. RQ2 — Error Geography

Validation errors:

**93/10500**
(**0.886%**)

Errors in the ten neurons with the largest raw error counts:

**41/93**
(**44.1%**)

### BMU distance

- Correct mean: 1.217743
- Error mean: 1.466475
- Correct median: 1.177700
- Error median: 1.393465

### Confidence

- Correct mean confidence: 0.997036
- Error mean confidence: 0.813575

### Exploratory neuron-level associations

- Error rate vs entropy:
  rho = 0.741762
- Error rate vs purity:
  rho = -0.736987
- Error rate vs mean BMU distance:
  rho = 0.349297

---

## 6. Main Week 3 Finding

The SOM exposes non-uniform error geography in the learned latent
representation.

Although the classifier makes relatively few validation errors,
44.1% of them occur in only ten SOM neurons when ranked by raw error
count.

Misclassified samples also have greater BMU distance on average.

However, neuron-level class entropy and purity are more strongly
associated with error rate than mean BMU distance in this development
run.

Therefore, the evidence supports structured SOM-based post-training
diagnosis but does not justify claiming that quantization distance
alone explains classifier failure.

---

## 7. Figures

RQ1:

1. rq1_train_hit_count
2. rq1_train_dominant_class
3. rq1_validation_purity
4. rq1_validation_entropy

RQ2:

5. rq2_validation_error_count
6. rq2_validation_error_rate
7. rq2_validation_wilson_lower95
8. rq2_validation_mean_confidence

All are saved as both PNG and PDF.

---

## 8. Integrity and Reproducibility

- train/validation-only SOM development
- no final-test use
- frozen grid-selection protocol
- frozen RQ1/RQ2 analysis protocol
- unique scientific run IDs
- duplicate-run protection
- model SHA-256 verification
- BMU assignment validation
- frozen QE/occupancy regression checks
- sample IDs retained
- input hashes retained
- output hashes retained
- failed attempt retained for provenance
- reusable unit tests

---

## 9. Limitations

- primary development comparison uses seed 42
- 15x15 was explored before formal grid-selection protocol
- sparse neurons can produce unstable local error rates
- correlations are exploratory
- no causal interpretation
- no novelty/OOD claim is made in Week 3

---

## 10. Conclusion

Week 3 successfully establishes a reproducible NNSOM post-training
analysis pipeline over the LeNet-5 MNIST representation.

The 20x20 map reveals strong class organization, localized overlap,
and non-uniform model error geography.

These results provide the required foundation for later novelty/OOD
and SOM-guided intervention experiments.

## Week 3 Status

**COMPLETE**
