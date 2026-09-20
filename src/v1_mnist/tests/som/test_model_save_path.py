"""Regression test for NNSOM model save-path handling."""

import os
import tempfile
import unittest
from pathlib import Path

from NNSOM.plots import SOMPlots


class TestNNSOMModelSavePath(unittest.TestCase):
    """Ensure NNSOM saves inside the requested model directory."""

    def test_trailing_separator_saves_model_inside_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "som_models"
            models_dir.mkdir(parents=True, exist_ok=True)

            model_name = "som_2x2_seed42"
            expected_path = models_dir / model_name

            som = SOMPlots(dimensions=(2, 2))

            som.save_pickle(
                model_name,
                str(models_dir) + os.sep,
            )

            self.assertTrue(expected_path.is_file())

            malformed_path = Path(
                str(models_dir) + model_name
            )
            self.assertFalse(malformed_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
