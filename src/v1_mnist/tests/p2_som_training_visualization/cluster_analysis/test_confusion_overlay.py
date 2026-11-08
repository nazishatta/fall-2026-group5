"""Confusion overlay on the SOM map: counts per neuron, ordering, figure and no-overwrite rule."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.v1_mnist.component.p2_som_training_visualization.cluster_analysis import confusion_overlay as co
from src.v1_mnist.component.utils.dataset_info import DatasetInfo

NUM_CLASSES = 3
GRID = 3
NUM_NEURONS = GRID * GRID

# validation images: (true class, predicted class, neuron)
IMAGES = [
    (0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 1), (0, 1, 2),  # 0 -> 1 : 3 errors
    (1, 0, 4), (1, 0, 4),                                    # 1 -> 0 : 2 errors
    (2, 1, 8), (2, 2, 8), (1, 1, 5), (2, 0, 6),              # 2 -> 1, 2 -> 0 : 1 error each
]


def make_assignments() -> co.Assignments:
    return co.Assignments(
        labels=np.array([i[0] for i in IMAGES]),
        predictions=np.array([i[1] for i in IMAGES]),
        neurons=np.array([i[2] for i in IMAGES]),
    )


def info() -> DatasetInfo:
    return DatasetInfo("toy", "Toy", NUM_CLASSES, "class", "Net", "fc", 4)


class TestPairs(unittest.TestCase):

    def test_pairs_are_ranked_by_errors_then_by_class(self) -> None:
        a = make_assignments()
        pairs = co.top_confusion_pairs(a.labels, a.predictions, 4)
        self.assertEqual(pairs, [(0, 1, 3), (1, 0, 2), (2, 0, 1), (2, 1, 1)])

    def test_number_of_pairs_is_limited_by_the_request(self) -> None:
        a = make_assignments()
        self.assertEqual(len(co.top_confusion_pairs(a.labels, a.predictions, 2)), 2)

    def test_neuron_counts_add_up_to_the_errors_of_the_pair(self) -> None:
        a = make_assignments()
        for true_class, predicted, errors in co.top_confusion_pairs(a.labels, a.predictions, 10):
            counts = co.pair_neuron_counts(a, true_class, predicted, NUM_NEURONS)
            self.assertEqual(len(counts), NUM_NEURONS)
            self.assertEqual(int(counts.sum()), errors)

    def test_counts_land_on_the_right_neurons(self) -> None:
        counts = co.pair_neuron_counts(make_assignments(), 0, 1, NUM_NEURONS)
        self.assertEqual(counts.tolist(), [0, 2, 1, 0, 0, 0, 0, 0, 0])

    def test_table_rows_add_up_to_the_errors(self) -> None:
        a = make_assignments()
        pairs = co.top_confusion_pairs(a.labels, a.predictions, 2)
        counts = [co.pair_neuron_counts(a, t, p, NUM_NEURONS) for t, p, _ in pairs]
        rows = co.overlay_rows(pairs, counts)
        for true_class, predicted, errors in pairs:
            total = sum(r["count"] for r in rows
                        if (r["true_class"], r["predicted_class"]) == (true_class, predicted))
            self.assertEqual(total, errors)


class TestChecks(unittest.TestCase):

    def test_hit_counts_must_match_the_cluster_table(self) -> None:
        a = make_assignments()
        support = np.bincount(a.neurons, minlength=NUM_NEURONS)
        co.check_against_table(a, support)
        wrong = support.copy()
        wrong[0] += 1
        with self.assertRaises(RuntimeError):
            co.check_against_table(a, wrong)

    def test_neuron_outside_the_map_is_refused(self) -> None:
        a = make_assignments()
        with self.assertRaises(RuntimeError):
            co.check_against_table(a, np.ones(4))

    def test_config_without_the_section_is_refused(self) -> None:
        class Empty:
            pass

        with self.assertRaises(KeyError):
            co.overlay_settings(Empty())


class TestFigureAndRun(unittest.TestCase):

    def test_figure_is_saved_as_pdf_only(self) -> None:
        a = make_assignments()
        pairs = co.top_confusion_pairs(a.labels, a.predictions, 3)
        counts = [co.pair_neuron_counts(a, t, p, NUM_NEURONS) for t, p, _ in pairs]
        xs, ys = np.meshgrid(np.arange(GRID, dtype=float), np.arange(GRID, dtype=float))
        coordinates = np.vstack([xs.ravel(), ys.ravel()])
        dominant = np.arange(NUM_NEURONS, dtype=float) % NUM_CLASSES
        fig = co.draw_overlay(coordinates, dominant, pairs, counts, info(), columns=2)
        with tempfile.TemporaryDirectory() as tmp:
            saved = co.save_pdf(fig, Path(tmp) / "figures" / co.FIGURE_NAME)
            self.assertEqual(saved.suffix, ".pdf")
            self.assertTrue(saved.read_bytes().startswith(b"%PDF"))
            self.assertEqual([p.suffix for p in saved.parent.parent.rglob("*") if p.is_file()], [".pdf"])

    def test_run_writes_files_and_never_overwrites(self) -> None:
        a = make_assignments()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "cluster" / "toy_model"
            (base / "assignments").mkdir(parents=True)
            (base / "tables").mkdir()
            (base / "cluster_summary.json").write_text(json.dumps({"test_used": False}))
            with (base / "assignments" / "val_bmu_assignments.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["label", "prediction", "bmu_index"])
                writer.writerows(zip(a.labels.tolist(), a.predictions.tolist(), a.neurons.tolist()))
            support = np.bincount(a.neurons, minlength=NUM_NEURONS)
            with (base / "tables" / "val_cluster_metrics.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["neuron_id", "pos_0", "pos_1", "support", "dominant_class"])
                for n in range(NUM_NEURONS):
                    writer.writerow([n, float(n % GRID), float(n // GRID), int(support[n]),
                                     n % NUM_CLASSES if support[n] else ""])
            config = root / "toy.yaml"
            config.write_text(
                "dataset: {name: toy, num_classes: 3}\n"
                "model: {architecture: Net, feature_layer: fc, feature_dim: 4}\n"
                f"paths: {{cluster_dir: '{(root / 'cluster').as_posix()}'}}\n"
                "visualization: {selected_model: toy_model}\n"
                "analysis: {confusion_overlay: {num_pairs: 2, panel_columns: 2}}\n"
            )
            with mock.patch.object(sys, "argv", ["confusion_overlay", "--config", str(config)]):
                co.main()
                self.assertTrue((base / "figures" / "pdf" / f"{co.FIGURE_NAME}.pdf").is_file())
                with (base / "tables" / co.TABLE_NAME).open(encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(sum(int(r["count"]) for r in rows), 3 + 2)
                with self.assertRaises(FileExistsError):
                    co.main()


if __name__ == "__main__":
    unittest.main()
