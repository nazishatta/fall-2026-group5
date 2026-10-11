"""Week 5 (Error Pattern Analysis): support-aware error-rate helpers.

Split from the former tests/som/test_bmu_rq_analysis.py.
"""

import math
import unittest

from src.v1_mnist.component.utils.stats import wilson_lower_bound


class TestErrorRateBounds(unittest.TestCase):

    def test_wilson_bound_is_valid(self) -> None:
        lower = wilson_lower_bound(
            errors=3,
            total=20,
        )

        self.assertGreaterEqual(
            lower,
            0.0,
        )

        self.assertLessEqual(
            lower,
            3 / 20,
        )

        self.assertTrue(
            math.isnan(
                wilson_lower_bound(
                    errors=0,
                    total=0,
                )
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
