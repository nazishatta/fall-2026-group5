"""The secondary SOM re-analysis config (max-softmax arm) differs from the primary one only where the addendum says."""

from __future__ import annotations

from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parents[5]
CONFIGS = CODE_ROOT / "src/v1_mnist/component/configs"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_only_the_run_name_arms_pairs_and_protocol_differ() -> None:
    primary, secondary = _load("som_reanalysis.yaml"), _load("som_reanalysis_msp_arm.yaml")
    assert secondary["run"]["name"] == "som_reanalysis_msp_arm" and secondary["run"]["revision"] == ""
    assert secondary["cells"]["arms"] == ["standard", "augmented_som", "augmented_msp"]
    assert secondary["comparison"]["pairs"] == [["augmented_som", "standard"], ["augmented_som", "augmented_msp"],
                                                ["augmented_msp", "standard"]]
    assert (CODE_ROOT / secondary["protocol"]["document"]).is_file()
    for key in ("inputs", "paths", "extraction", "measures", "figures"):
        assert secondary[key] == primary[key], key
    assert secondary["cells"]["seeds"] == primary["cells"]["seeds"]
    assert secondary["comparison"]["confidence"] == primary["comparison"]["confidence"]
    assert secondary["comparison"]["organisation_tolerance_pct"] == primary["comparison"]["organisation_tolerance_pct"]
