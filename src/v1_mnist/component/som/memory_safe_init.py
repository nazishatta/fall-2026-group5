"""Memory-safe initialization wrapper for NNSOM.

Applies reduced/economy SVD to NumPy and, when installed, CuPy.
This wrapper does not select the SOM backend.
"""

from contextlib import contextmanager
from typing import Iterator

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


@contextmanager
def economy_svd_for_nnsom_init() -> Iterator[None]:
    """Use reduced SVD during NNSOM initialization."""

    original_numpy_svd = np.linalg.svd

    def economy_numpy_svd(
        a,
        *args,
        **kwargs,
    ):
        if kwargs.get("full_matrices") is True:
            raise RuntimeError(
                "Unexpected explicit full_matrices=True "
                "during NNSOM initialization."
            )

        kwargs["full_matrices"] = False

        return original_numpy_svd(
            a,
            *args,
            **kwargs,
        )

    np.linalg.svd = economy_numpy_svd

    original_cupy_svd = None

    if cp is not None:
        original_cupy_svd = cp.linalg.svd

        def economy_cupy_svd(
            a,
            *args,
            **kwargs,
        ):
            if kwargs.get("full_matrices") is True:
                raise RuntimeError(
                    "Unexpected explicit full_matrices=True "
                    "during NNSOM initialization."
                )

            kwargs["full_matrices"] = False

            return original_cupy_svd(
                a,
                *args,
                **kwargs,
            )

        cp.linalg.svd = economy_cupy_svd

    try:
        yield

    finally:
        np.linalg.svd = original_numpy_svd

        if (
            cp is not None
            and original_cupy_svd is not None
        ):
            cp.linalg.svd = original_cupy_svd
