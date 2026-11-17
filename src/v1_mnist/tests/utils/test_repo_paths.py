"""Tests for the shared repository path check."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.v1_mnist.component.utils.repo_paths import resolve_inside


def test_relative_path_resolves_inside_root(tmp_path: Path) -> None:
    """A normal config path resolves under the root."""
    assert (
        resolve_inside(tmp_path, "outputs/run/file.json")
        == tmp_path / "outputs/run/file.json"
    )
    assert resolve_inside(tmp_path, "a/../b.txt") == tmp_path / "b.txt"


@pytest.mark.parametrize("value", ["../outside", "a/../../outside", "/etc/passwd"])
def test_paths_leaving_root_are_refused(tmp_path: Path, value: str) -> None:
    """Parent-directory and absolute paths outside the root raise ValueError."""
    with pytest.raises(ValueError, match="outside repository"):
        resolve_inside(tmp_path, value)


def test_linked_data_file_is_accepted(tmp_path: Path) -> None:
    """A symlink inside the root that points elsewhere is allowed (data linked into the repo)."""
    outside = tmp_path / "elsewhere" / "test_32x32.mat"
    outside.parent.mkdir()
    outside.write_bytes(b"svhn")
    root = tmp_path / "repo"
    (root / "data" / "ood_svhn").mkdir(parents=True)
    (root / "data" / "ood_svhn" / "test_32x32.mat").symlink_to(outside)
    path = resolve_inside(root, "data/ood_svhn/test_32x32.mat")
    assert path == root / "data/ood_svhn/test_32x32.mat"
    assert path.read_bytes() == b"svhn"
