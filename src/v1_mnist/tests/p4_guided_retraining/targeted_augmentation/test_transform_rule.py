"""The transform-check rule in code gives what Fashion-MNIST did by hand (docs/DEVIATIONS.md, 2026-11-21)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

pytest.importorskip("torch")

from src.v1_mnist.component.p4_guided_retraining.targeted_augmentation.transform_rule import (  # noqa: E402
    accepted_transforms,
    attempts_until_pass,
    check_path,
    halved,
    max_halvings,
)

ROOT = Path(__file__).resolve().parents[5]
MNIST = ROOT / "src/v1_mnist/component/configs/retraining.yaml"
FASHION = ROOT / "src/v1_fashion_mnist/component/configs/retraining.yaml"
CHECKS = ROOT / "outputs/v1_fashion_mnist/p4_guided_retraining/targeted_augmentation"


def _transforms(config: Path) -> dict[str, float]:
    return yaml.safe_load(config.read_text(encoding="utf-8"))["augmentation"]["transforms"]


def _settings(tmp_path: Path, rule: dict[str, Any] | None) -> Any:
    options = {"transforms": _transforms(MNIST), **({"transform_check": rule} if rule else {})}
    return SimpleNamespace(options=options, output_dir=tmp_path)


def test_halving_twice_gives_the_ranges_fashion_mnist_uses() -> None:
    assert halved(_transforms(MNIST), 0) == pytest.approx(_transforms(MNIST))
    assert halved(_transforms(MNIST), 1) == pytest.approx(
        {"max_rotation_degrees": 5.0, "max_shift_pixels": 0.5, "min_zoom": 0.95, "max_zoom": 1.05, "max_shear_degrees": 4.0})
    assert halved(_transforms(MNIST), 2) == pytest.approx(_transforms(FASHION))


def test_the_recorded_fashion_mnist_checks_lead_the_rule_to_r3() -> None:
    """Fed the drops measured in r1, r2 and r3, the rule stops at the second halving, as done by hand."""
    recorded = [json.loads((CHECKS / f"augmentation_comparison_som_15x15_nb11_ep250_s42{suffix}" / "check" /
                            "augmentation_check.json").read_text(encoding="utf-8")) for suffix in ("", "_r2", "_r3")]
    drops = {json.dumps(halved(r["transforms"], 0), sort_keys=True): r["accuracy_drop"] for r in recorded}
    attempts = attempts_until_pass(_transforms(MNIST), 0.02, 2, lambda t: drops[json.dumps(t, sort_keys=True)])
    assert [a["passed"] for a in attempts] == [False, False, True]
    assert attempts[-1]["transforms"] == pytest.approx(_transforms(FASHION))


def test_halving_stops_at_the_first_pass_and_after_the_allowed_number() -> None:
    drops = iter([0.01])
    assert len(attempts_until_pass(_transforms(MNIST), 0.02, 2, lambda _t: next(drops))) == 1
    drops = iter([0.08, 0.03])
    attempts = attempts_until_pass(_transforms(MNIST), 0.02, 1, lambda _t: next(drops))
    assert [a["halvings"] for a in attempts] == [0, 1] and attempts[-1]["passed"] is False
    assert len(attempts_until_pass(_transforms(MNIST), None, 2, lambda _t: 0.5)) == 1   # no limit: measure only


def test_training_uses_the_config_ranges_unless_halving_is_allowed(tmp_path: Path) -> None:
    assert max_halvings(_settings(tmp_path, None)) == 0
    assert accepted_transforms(_settings(tmp_path, None)) == _transforms(MNIST)
    assert accepted_transforms(_settings(tmp_path, {"max_accuracy_drop": 0.02})) == _transforms(MNIST)
    settings = _settings(tmp_path, {"max_accuracy_drop": 0.02, "max_halvings": 2})
    with pytest.raises(RuntimeError, match="run --stage check first"):
        accepted_transforms(settings)
    check_path(settings).parent.mkdir(parents=True)
    check_path(settings).write_text(json.dumps({"transforms": halved(_transforms(MNIST), 2)}))
    assert accepted_transforms(settings) == pytest.approx(_transforms(FASHION))


def test_the_current_configs_do_not_use_automatic_halving(tmp_path: Path) -> None:
    """MNIST and Fashion-MNIST ran before the rule was in the code; their configs are unchanged."""
    for config in (MNIST, FASHION):
        block = yaml.safe_load(config.read_text(encoding="utf-8"))["augmentation"]
        assert "max_halvings" not in (block.get("transform_check") or {})


def test_the_check_stage_halves_records_every_attempt_and_hands_the_ranges_to_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_check end to end on made-up data, with the network's answers scripted per attempt."""
    import numpy as np
    import torch

    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.lenet5 import LeNet5
    from src.v1_mnist.component.p4_guided_retraining.targeted_augmentation import augmentation_check
    from src.v1_mnist.tests.p4_guided_retraining.synthetic_data import NUM_CLASSES, make_data

    data = make_data()
    checkpoint = tmp_path / "lenet.pth"
    torch.save(LeNet5(num_classes=NUM_CLASSES).state_dict(), checkpoint)
    labels = data.val.labels
    wrong = (labels + 1) % NUM_CLASSES
    # Originals all right; altered images wrong for 9, then 3, then 0 of the 30 (drops 0.3, 0.1, 0.0).
    answers = iter([labels, np.where(np.arange(30) < 9, wrong, labels), np.where(np.arange(30) < 3, wrong, labels),
                    labels])
    monkeypatch.setattr(augmentation_check, "predict", lambda *_args: (next(answers), None))
    settings = SimpleNamespace(
        options={"transforms": _transforms(MNIST), "copies_per_image": 3,
                 "transform_check": {"max_accuracy_drop": 0.05, "max_halvings": 2}},
        output_dir=tmp_path / "run", run_id="test_run", cnn_checkpoint=checkpoint, device="cpu", seeds=[1],
        train=SimpleNamespace(num_classes=NUM_CLASSES, in_channels=1, image_size=28),
    )
    result = augmentation_check.run_check(settings, data)
    assert [a["halvings"] for a in result["attempts"]] == [0, 1, 2]
    assert result["passed"] is True and result["halvings"] == 2
    assert result["accuracy_drop"] == pytest.approx(0.0)
    assert result["transforms"] == pytest.approx(halved(_transforms(MNIST), 2))
    assert (settings.output_dir / "figures" / "pdf" / "augmentation_contact_sheet.pdf").is_file()
    assert accepted_transforms(settings) == pytest.approx(halved(_transforms(MNIST), 2))
    augmentation_check.require_passed_check(settings)
