# Phase 3: Novelty Detection Tests

The Phase 3 test directory is no longer empty.

Test coverage includes novelty-score calculations, OOD comparison helpers,
and the standalone Phase 3 pipeline adapter.

## Verified EC2 checks

- Complete repository suite: **168 passed, 2 warnings, 0 failed**
  on the compatible EC2 environment.
- Four standalone Phase 3 integration tests passed.
- Fashion-MNIST baseline reproduction tests passed.
- Independent numerical reproduction confirmed the five Fashion-MNIST
  methods: QE, max-softmax, kNN, energy, and Mahalanobis.
- Ruff checks passed across Phase 3 scoring, pipeline, and tests.
- The two warnings originate from existing Seaborn and NNSOM code.

## Environment and reproducibility limitations

The local Intel macOS environment previously could not collect the
complete suite using the repository's pinned PyTorch version.
The full suite was subsequently verified on EC2; the earlier local
environment limitation should not be confused with a current EC2
test failure.

The verified suite establishes code-test success in that environment.
It does not establish cross-platform numerical equivalence,
multiple independent model-training seeds, or resolution of every
historical baseline concern.

The standalone Phase 3 adapter is tested independently; shared
pipeline registration is intentionally deferred to another PR.
