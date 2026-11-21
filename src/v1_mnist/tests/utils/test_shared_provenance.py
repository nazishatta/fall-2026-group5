"""Check consistency of the frozen Phase 3 provenance manifest."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "component" / "configs"

CONFIG_NAMES = (
    "required_ood_baselines.yaml",
    "svhn_knn_baselines.yaml",
    "svhn_paired_comparison.yaml",
    "svhn_score_export.yaml",
    "autoencoder_standardized_diagnostic.yaml",
)


def test_phase3_configs_match_shared_provenance() -> None:
    """All five Phase 3 configs must agree with frozen input fingerprints."""
    shared = yaml.safe_load(
        (CONFIG_DIR / "phase3_shared_provenance.yaml").read_text()
    )
    for name in CONFIG_NAMES:
        config = yaml.safe_load((CONFIG_DIR / name).read_text())
        for key in ("checkpoint", "train_features", "validation_features"):
            assert config["inputs"][key] == shared["inputs"][key], name
            assert (
                config["expected_sha256"][key]
                == shared["expected_sha256"][key]
            ), name
        assert config["inputs"]["svhn_file"] == shared["svhn_file"], name


# ---- one manifest per dataset folder --------------------------------------------------


def test_manifest_for_mnist_configs_is_the_mnist_manifest() -> None:
    from src.v1_mnist.component.utils.shared_provenance import MANIFEST, manifest_path

    assert manifest_path() == MANIFEST
    assert manifest_path("src/v1_mnist/component/configs/qe_ood_evaluation.yaml") == MANIFEST


def test_manifest_for_a_folder_without_base_yaml_is_the_mnist_manifest(tmp_path) -> None:
    from src.v1_mnist.component.utils.shared_provenance import MANIFEST, manifest_path

    assert manifest_path(tmp_path / "step.yaml") == MANIFEST


def test_manifest_is_read_from_the_dataset_folder(tmp_path) -> None:
    from src.v1_mnist.component.utils.shared_provenance import (
        MANIFEST_NAME,
        load_manifest,
    )

    (tmp_path / "base.yaml").write_text("dataset:\n  name: fashion_mnist\n", encoding="utf-8")
    (tmp_path / MANIFEST_NAME).write_text("svhn_file: data/x.mat\n", encoding="utf-8")
    assert load_manifest(tmp_path / "step.yaml") == {"svhn_file": "data/x.mat"}


def test_dataset_folder_without_its_own_manifest_stops(tmp_path) -> None:
    import pytest

    from src.v1_mnist.component.utils.shared_provenance import manifest_path

    (tmp_path / "base.yaml").write_text("dataset:\n  name: fashion_mnist\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="needs its own Phase 3 manifest"):
        manifest_path(tmp_path / "step.yaml")
