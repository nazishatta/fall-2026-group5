"""Week 1 (Data Pipeline): class-distribution figure is saved as one PDF (no SVG).

Split from the former tests/som/test_figure_dual_format.py.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")

from src.v1_mnist.component.p1_model_training_baseline.data_pipeline.data_plots import plot_class_distribution


class TestClassDistributionFigure(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_plots_helper_creates_pdf_only(self) -> None:
        counts = {0: {"count": 10}, 1: {"count": 15}}
        target = self.root / "train_dist"
        plot_class_distribution(counts, "Train Dist", target)

        expected_pdf = self.root / "pdf" / "train_dist.pdf"

        self.assertTrue(expected_pdf.is_file())
        self.assertFalse((self.root / "svg").exists())
        self.assertNotIn(b"/Type3", expected_pdf.read_bytes())


if __name__ == "__main__":
    unittest.main()
