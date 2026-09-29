"""Small NumPy helpers: CuPy-safe conversion and JSON-friendly rounding."""

from __future__ import annotations

import math

import numpy as np


def to_numpy(a) -> np.ndarray:
    """Return a NumPy array even if NNSOM handed back CuPy arrays."""

    if hasattr(a, "get"):
        a = a.get()
    return np.asarray(a)


def round_sig(arr: np.ndarray, sig: int) -> np.ndarray:
    """Round to `sig` significant digits (keeps tiny activations visible)."""

    arr = np.asarray(arr, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.floor(np.log10(np.abs(arr)))
    mag = np.where(np.isfinite(mag), mag, 0)
    factor = 10.0 ** (sig - 1 - mag)
    return np.round(arr * factor) / factor


def clean_values(values, decimals: int = 4, sig: int | None = None):
    """Round floats and turn NaN/inf into None so json.dumps writes null.

    Use ``sig`` (significant digits) for raw fc2 activations: some
    dimensions only span ~1e-4, which fixed-decimal rounding would erase.
    """

    arr = np.asarray(values, dtype=np.float64)
    arr = round_sig(arr, sig) if sig else np.round(arr, decimals)
    out = arr.tolist()

    def fix(v):
        if isinstance(v, list):
            return [fix(x) for x in v]
        return None if (v is None or not math.isfinite(v)) else v

    return fix(out)
