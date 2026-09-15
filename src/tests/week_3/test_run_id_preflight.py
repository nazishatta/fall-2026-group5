"""Regression tests for Week 3 scientific run output isolation."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_PATH = (
    REPO_ROOT
    / "src"
    / "component"
    / "pipeline"
    / "04_train_som.py"
)

SPEC = importlib.util.spec_from_file_location(
    "week3_train_som_pipeline",
    PIPELINE_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise RuntimeError(
        f"Could not load Week 3 SOM pipeline from {PIPELINE_PATH}"
    )

PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)

preflight_scientific_outputs = (
    PIPELINE.preflight_scientific_outputs
)


class TestScientificRunPreflight(unittest.TestCase):
    """Ensure duplicate run IDs fail before expensive SOM training."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

        root = Path(self.tempdir.name)

        self.logs_dir = root / "logs"
        self.models_dir = root / "som_models"

        self.logs_dir.mkdir(parents=True)
        self.models_dir.mkdir(parents=True)

        self.run_id = "som_10x10_seed42_gridstudy"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_new_run_returns_expected_paths(self):
        metrics_path, model_path = preflight_scientific_outputs(
            run_id=self.run_id,
            logs_dir=self.logs_dir,
            models_dir=self.models_dir,
        )

        self.assertEqual(
            metrics_path,
            self.logs_dir / f"{self.run_id}_metrics.json",
        )

        self.assertEqual(
            model_path,
            self.models_dir / self.run_id,
        )

    def test_existing_metrics_reject_duplicate_run(self):
        metrics_path = (
            self.logs_dir / f"{self.run_id}_metrics.json"
        )

        metrics_path.write_text(
            "{}",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            FileExistsError,
            self.run_id,
        ):
            preflight_scientific_outputs(
                run_id=self.run_id,
                logs_dir=self.logs_dir,
                models_dir=self.models_dir,
            )

    def test_existing_model_reject_duplicate_run(self):
        model_path = self.models_dir / self.run_id

        model_path.write_bytes(b"existing-model")

        with self.assertRaisesRegex(
            FileExistsError,
            self.run_id,
        ):
            preflight_scientific_outputs(
                run_id=self.run_id,
                logs_dir=self.logs_dir,
                models_dir=self.models_dir,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
