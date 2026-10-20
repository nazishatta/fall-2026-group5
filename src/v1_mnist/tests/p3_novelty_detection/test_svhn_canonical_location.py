"""Check canonical SVHN dataset location without loading datasets."""

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.shared_provenance import (
    ROOT,
    load_manifest,
    verify_shared_inputs,
)


def test_canonical_svhn_dataset_path() -> None:
    """Canonical manifest points to the frozen SVHN test file."""
    relative = str(load_manifest()["svhn_file"])
    assert relative == "data/ood_svhn/test_32x32.mat"
    assert (ROOT / relative).is_file()
    verify_shared_inputs(
        {"svhn_file": relative}, required_keys=("svhn_file",)
    )
