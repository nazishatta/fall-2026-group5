"""Tests verifying dual-format SVG and PDF figure outputs in dedicated directories."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.v1_mnist.component.analysis.plotting import (
    matrix_from_neuron_values,
    save_heatmap,
)
from src.v1_mnist.component.pipeline.train_som import save_qe_history
from src.v1_mnist.component.visualization.plots import plot_class_distribution
from src.v1_mnist.component.visualization.som_visualizations import (
    _route_dual_format_paths,
    save_figure,
)


class DummyQEHistory:
    def __init__(self):
        self.epoch = [1, 2, 3]
        self.quantization_error = [1.5, 1.2, 0.9]
        self.neighborhood_radius = [3.0, 2.0, 1.0]

    def to_dict(self):
        return {
            "epoch": self.epoch,
            "quantization_error": self.quantization_error,
            "neighborhood_radius": self.neighborhood_radius,
        }


class TestFigureDualFormatRouting(unittest.TestCase):
    """Ensure all plotting helpers save to dedicated svg/ and pdf/ directories."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_route_dual_format_paths_categories(self):
        categories = [
            "core",
            "class_maps",
            "feature_maps",
            "component_planes",
            "analysis_maps",
            "extended",
        ]
        for cat in categories:
            dummy_path = self.root / cat / "test_fig"
            svg_p, pdf_p = _route_dual_format_paths(dummy_path)
            self.assertEqual(svg_p, self.root / "svg" / cat / "test_fig.svg")
            self.assertEqual(pdf_p, self.root / "pdf" / cat / "test_fig.pdf")

    def test_save_figure_creates_svg_and_pdf_in_mirror_folders(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        out_base = self.root / "core" / "topology"
        saved = save_figure(fig, out_base)

        expected_svg = self.root / "svg" / "core" / "topology.svg"
        expected_pdf = self.root / "pdf" / "core" / "topology.pdf"

        self.assertTrue(expected_svg.is_file(), f"Missing {expected_svg}")
        self.assertTrue(expected_pdf.is_file(), f"Missing {expected_pdf}")
        self.assertIn(expected_svg, saved)
        self.assertIn(expected_pdf, saved)

        # Ensure no empty bare 'core' directory was created directly under root
        self.assertFalse(
            (self.root / "core").exists(),
            "Bare uncategorized 'core' directory should not be created at root.",
        )

    def test_plots_helper_creates_svg_and_pdf(self):
        counts = {0: {"count": 10}, 1: {"count": 15}}
        target = self.root / "train_dist"
        plot_class_distribution(counts, "Train Dist", target)

        expected_svg = self.root / "svg" / "train_dist.svg"
        expected_pdf = self.root / "pdf" / "train_dist.pdf"

        self.assertTrue(expected_svg.is_file())
        self.assertTrue(expected_pdf.is_file())

    def test_train_som_save_qe_history_creates_svg_and_pdf(self):
        history = DummyQEHistory()
        artifacts = save_qe_history(
            history=history,
            run_id="test_run",
            logs_dir=self.root / "logs",
            tables_dir=self.root / "tables",
            figures_dir=self.root / "figures",
        )

        expected_svg = self.root / "figures" / "svg" / "convergence" / "test_run_qe_curve.svg"
        expected_pdf = self.root / "figures" / "pdf" / "convergence" / "test_run_qe_curve.pdf"

        self.assertTrue(expected_svg.is_file())
        self.assertTrue(expected_pdf.is_file())
        self.assertEqual(artifacts["figure_svg"], str(expected_svg))
        self.assertEqual(artifacts["figure_pdf"], str(expected_pdf))

    def test_save_heatmap_creates_svg_and_pdf(self):
        coords = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        matrix = matrix_from_neuron_values(coords, vals)

        saved = save_heatmap(
            path_base=self.root / "rq1_hit_count",
            matrix=matrix,
            title="RQ1 Test",
            colorbar_label="count",
        )

        expected_svg = self.root / "svg" / "rq1_hit_count.svg"
        expected_pdf = self.root / "pdf" / "rq1_hit_count.pdf"

        self.assertTrue(expected_svg.is_file())
        self.assertTrue(expected_pdf.is_file())
        self.assertIn(expected_svg, saved)
        self.assertIn(expected_pdf, saved)


if __name__ == "__main__":
    unittest.main()
