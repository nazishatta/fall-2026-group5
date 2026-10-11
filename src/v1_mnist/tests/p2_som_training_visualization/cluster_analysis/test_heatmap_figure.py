"""Week 4 (Cluster Analysis): SOM heatmaps are saved as one PDF (no SVG).

Split from the former tests/som/test_figure_dual_format.py.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np

from src.v1_mnist.component.utils.heatmaps import (
    matrix_from_neuron_values,
    save_heatmap,
)


class TestHeatmapFigure(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_save_heatmap_creates_pdf_only(self) -> None:
        coords = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        matrix = matrix_from_neuron_values(coords, vals)

        saved = save_heatmap(
            path_base=self.root / "hit_count",
            matrix=matrix,
            title="Hit Count Test",
            colorbar_label="count",
        )

        expected_pdf = self.root / "pdf" / "hit_count.pdf"

        self.assertTrue(expected_pdf.is_file())
        self.assertEqual(saved, [expected_pdf])
        self.assertFalse((self.root / "svg").exists())
        self.assertNotIn(b"/Type3", expected_pdf.read_bytes())


if __name__ == "__main__":
    unittest.main()


def test_class_map_works_for_more_than_ten_classes(tmp_path: Path) -> None:
    import numpy as np
    from src.v1_mnist.component.utils.heatmaps import class_colormap_name, matrix_from_neuron_values, save_heatmap

    assert class_colormap_name(10) == "tab10" and class_colormap_name(15) == "tab20"
    coords = np.vstack([np.repeat(np.arange(4), 4), np.tile(np.arange(4), 4)]).astype(float)
    values = np.arange(16) % 15
    paths = save_heatmap(tmp_path / "dominant", matrix_from_neuron_values(coords, values),
                         "Dominant class", "class", categorical=True, num_classes=15)
    assert all(p.exists() for p in paths)


def test_class_map_needs_the_class_count(tmp_path: Path) -> None:
    import numpy as np
    import pytest
    from src.v1_mnist.component.utils.heatmaps import matrix_from_neuron_values, save_heatmap

    coords = np.vstack([np.arange(4.0), np.zeros(4)])
    with pytest.raises(ValueError, match="num_classes"):
        save_heatmap(tmp_path / "x", matrix_from_neuron_values(coords, np.arange(4.0)), "t", "l", categorical=True)
