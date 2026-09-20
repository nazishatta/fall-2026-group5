"""Tests for Week 3 BMU/RQ analysis helpers."""

import math
import unittest

import numpy as np

from src.component.analysis.som_bmu_rq_analysis import (
    matrix_from_neuron_values,
    normalized_entropy,
    reconstruct_assignments,
    wilson_lower_bound,
)


class TestBMURQAnalysis(unittest.TestCase):

    def test_entropy_pure_neuron_is_zero(self):
        counts = np.zeros(10, dtype=int)
        counts[3] = 25

        self.assertAlmostEqual(
            normalized_entropy(counts),
            0.0,
            places=12,
        )

    def test_entropy_uniform_neuron_is_one(self):
        counts = np.ones(10, dtype=int)

        self.assertAlmostEqual(
            normalized_entropy(counts),
            1.0,
            places=12,
        )

    def test_wilson_bound_is_valid(self):
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

    def test_native_coordinate_payload_allows_staggered_grid(self):
        coordinates = np.zeros((2, 400), dtype=float)

        # Mimic a staggered topology: alternating rows use
        # integer and half-integer x positions. Across the
        # full 20x20 map this produces 40 unique x values.
        neuron = 0

        for row in range(20):
            offset = 0.5 if row % 2 else 0.0

            for column in range(20):
                coordinates[0, neuron] = (
                    column + offset
                )
                coordinates[1, neuron] = row
                neuron += 1

        values = np.arange(
            400,
            dtype=float,
        )

        returned_coordinates, returned_values = (
            matrix_from_neuron_values(
                coordinates,
                values,
            )
        )

        self.assertEqual(
            len(
                np.unique(
                    returned_coordinates[0]
                )
            ),
            40,
        )

        self.assertEqual(
            len(
                np.unique(
                    returned_coordinates.T,
                    axis=0,
                )
            ),
            400,
        )

        np.testing.assert_array_equal(
            returned_values,
            values,
        )


    def test_reconstruct_assignments(self):
        clusters = [
            np.array([0, 2]),
            np.array([1]),
        ] + [
            np.array(
                [],
                dtype=int,
            )
            for _ in range(398)
        ]

        distances = [
            np.array([0.1, 0.3]),
            np.array([0.2]),
        ] + [
            np.array(
                [],
                dtype=float,
            )
            for _ in range(398)
        ]

        bmu, dist = reconstruct_assignments(
            clusters,
            distances,
            n_samples=3,
        )

        np.testing.assert_array_equal(
            bmu,
            np.array([0, 1, 0]),
        )

        np.testing.assert_allclose(
            dist,
            np.array([0.1, 0.2, 0.3]),
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
