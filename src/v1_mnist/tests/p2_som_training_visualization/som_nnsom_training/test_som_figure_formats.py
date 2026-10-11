"""Week 3 (SOM Setup with NNSOM): figures are saved as one PDF each, with embedded
TrueType fonts (pdf.fonttype 42), and dense plots rasterize their data layer only.

Split from the former tests/som/test_figure_dual_format.py.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.run_records import save_qe_history
from src.v1_mnist.component.utils.figures import pdf_figure_path, save_pdf
from src.v1_mnist.component.utils.som_plotting import save_figure


class DummyQEHistory:
    def __init__(self) -> None:
        self.epoch = [1, 2, 3]
        self.quantization_error = [1.5, 1.2, 0.9]
        self.neighborhood_radius = [3.0, 2.0, 1.0]

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "quantization_error": self.quantization_error,
            "neighborhood_radius": self.neighborhood_radius,
        }


def dense_scatter_figure() -> plt.Figure:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    ax.scatter(rng.normal(size=20000), rng.normal(size=20000), s=2)
    ax.set_title("Dense scatter")
    ax.set_xlabel("x")
    return fig


class TestSOMFigureFormats(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_pdf_figure_path_categories(self) -> None:
        for cat in ["core", "class_maps", "feature_maps", "component_planes", "analysis_maps", "extended"]:
            self.assertEqual(pdf_figure_path(self.root / cat / "test_fig"), self.root / "pdf" / cat / "test_fig.pdf")
        # Older callers that still name an svg/ path or an extension get the PDF location.
        self.assertEqual(pdf_figure_path(self.root / "svg" / "core" / "x.svg"), self.root / "pdf" / "core" / "x.pdf")
        self.assertEqual(pdf_figure_path(self.root / "plot.png"), self.root / "pdf" / "plot.pdf")

    def test_save_figure_creates_one_pdf_with_truetype_fonts(self) -> None:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_title("Topology")

        saved = save_figure(fig, self.root / "core" / "topology")

        expected_pdf = self.root / "pdf" / "core" / "topology.pdf"
        self.assertEqual(saved, [expected_pdf])
        data = expected_pdf.read_bytes()
        self.assertNotIn(b"/Type3", data)
        self.assertIn(b"/CIDFontType2", data)    # TrueType font (pdf.fonttype 42)
        self.assertFalse((self.root / "svg").exists())
        self.assertFalse((self.root / "core").exists(), "Bare 'core' folder should not be created.")

    def test_dense_layer_is_rasterized_but_text_stays_vector(self) -> None:
        vector = save_pdf(dense_scatter_figure(), self.root / "vector")
        raster = save_pdf(dense_scatter_figure(), self.root / "raster", rasterize_dense=True)
        raster_data = raster.read_bytes()
        self.assertIn(b"/Image", raster_data)          # the points became one image
        self.assertIn(b"/CIDFontType2", raster_data)   # the title and labels are still text
        self.assertLess(raster.stat().st_size, vector.stat().st_size / 3)

    def test_train_som_save_qe_history_creates_pdf_only(self) -> None:
        artifacts = save_qe_history(
            history=DummyQEHistory(),
            run_id="test_run",
            logs_dir=self.root / "logs",
            tables_dir=self.root / "tables",
            figures_dir=self.root / "figures",
        )
        expected_pdf = self.root / "figures" / "pdf" / "convergence" / "test_run_qe_curve.pdf"
        self.assertTrue(expected_pdf.is_file())
        self.assertEqual(artifacts["figure_pdf"], str(expected_pdf))
        self.assertNotIn("figure_svg", artifacts)
        self.assertFalse((self.root / "figures" / "svg").exists())


if __name__ == "__main__":
    unittest.main()
