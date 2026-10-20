# Phase 3: Novelty Detection Tests

The Phase 3 test directory is no longer empty.

Test coverage includes novelty-score calculations, OOD comparison helpers,
and the standalone Phase 3 pipeline adapter.

## Verified EC2 checks

- Complete repository suite: **168 passed, 2 warnings, 0 failed**
  on the compatible EC2 environment.
- Four standalone Phase 3 integration tests passed.
- The SVHN-only OOD configuration and runner tests passed (10 targeted tests).
- A complete test-suite rerun is still required after the scope cleanup.
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

## Test-count and benchmark provenance clarification

The earlier full-repository count of 168 passed and 2 warnings is
a historical EC2 verification result. The latest Phase 3-only suite,
executed on 2026-10-20, passed 134 tests with no failures.
These figures describe different test scopes and must not be treated
as conflicting measurements of the same test suite.

The consolidated nine-method CSV contains results from two source runs,
not a newly fitted single execution. It preserves each method's
source_run_id. QE and U-Matrix were evaluated together in the original
five-method benchmark; max-softmax, energy, Mahalanobis and kNN were
evaluated in the subsequent frozen-CNN baseline run.

The equal-fitting-budget diagnostic uses five explicit seeds, stored
under `seed_list` in its original configuration. That original
configuration is retained because its SHA-256 is recorded in the
committed experiment metrics. This is a naming compatibility exception
to the requested `seeds` key, not evidence that the experiment was
executed without a seed list.
