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
