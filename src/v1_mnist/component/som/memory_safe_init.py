"""Memory-safe initialization wrapper for NNSOM 1.8.3.

NNSOM 1.8.3 calls np.linalg.svd(xc) with NumPy's default
full_matrices=True inside SOM.init_w(). For large training sets this
can allocate an unnecessary O(n_samples^2) matrix.

This wrapper changes only that SVD call to full_matrices=False while
delegating all remaining initialization logic to NNSOM unchanged.
"""

from contextlib import contextmanager
from typing import Iterator

import numpy as np


@contextmanager
def economy_svd_for_nnsom_init() -> Iterator[None]:
    """Temporarily force NumPy SVD to use economy/reduced matrices."""
    original_svd = np.linalg.svd

    def economy_svd(a, *args, **kwargs):
        if kwargs.get("full_matrices") is True:
            raise RuntimeError(
                "Unexpected explicit full_matrices=True encountered "
                "while applying NNSOM memory-safe initialization."
            )

        kwargs["full_matrices"] = False
        return original_svd(a, *args, **kwargs)

    np.linalg.svd = economy_svd

    try:
        yield
    finally:
        np.linalg.svd = original_svd
