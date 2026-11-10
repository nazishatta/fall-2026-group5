"""Tests for the shared repository path check."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.v1_mnist.component.utils.repo_paths import resolve_inside


def test_relative_path_resolves_inside_root(tmp_path: Path) -> None:
    """A normal config path resolves under the root."""
    assert (
        resolve_inside(tmp_path, "outputs/run/file.json")
        == (tmp_path / "outputs/run/file.json").resolve()
    )


@pytest.mark.parametrize("value", ["../outside", "a/../../outside", "/etc/passwd"])
def test_paths_leaving_root_are_refused(tmp_path: Path, value: str) -> None:
    """Parent-directory and absolute paths outside the root raise ValueError."""
    with pytest.raises(ValueError, match="outside repository"):
        resolve_inside(tmp_path, value)
