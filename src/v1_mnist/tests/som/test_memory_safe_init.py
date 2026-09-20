"""Regression tests for the NNSOM memory-safe initialization workaround."""

import unittest

import numpy as np
from NNSOM.plots import SOMPlots

from src.component.som.memory_safe_init import economy_svd_for_nnsom_init


class TestEconomySVDForNNSOMInit(unittest.TestCase):
    """Verify that the workaround is behavior-preserving and safely scoped."""

    def test_numpy_svd_is_restored_after_context(self):
        """The global NumPy SVD function must always be restored."""
        original_svd = np.linalg.svd

        with economy_svd_for_nnsom_init():
            self.assertIsNot(np.linalg.svd, original_svd)

        self.assertIs(np.linalg.svd, original_svd)

    def test_numpy_svd_is_restored_after_exception(self):
        """The original SVD must also be restored if initialization fails."""
        original_svd = np.linalg.svd

        with self.assertRaises(RuntimeError):
            with economy_svd_for_nnsom_init():
                self.assertIsNot(np.linalg.svd, original_svd)
                raise RuntimeError("intentional regression-test exception")

        self.assertIs(np.linalg.svd, original_svd)

    def test_economy_initializer_matches_original_initializer(self):
        """Reduced SVD must preserve NNSOM initialization weights."""
        rng = np.random.default_rng(12345)
        x = rng.normal(size=(64, 8)).astype(np.float64)

        np.random.seed(42)
        original_som = SOMPlots(dimensions=(4, 4))
        original_som.init_w(x, norm_func=None)
        original_weights = np.array(original_som.w, copy=True)

        np.random.seed(42)
        economy_som = SOMPlots(dimensions=(4, 4))

        with economy_svd_for_nnsom_init():
            economy_som.init_w(x, norm_func=None)

        economy_weights = np.array(economy_som.w, copy=True)

        self.assertEqual(original_weights.shape, economy_weights.shape)

        np.testing.assert_allclose(
            economy_weights,
            original_weights,
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
