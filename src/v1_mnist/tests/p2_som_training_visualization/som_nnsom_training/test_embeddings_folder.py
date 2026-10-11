"""SOM code reads embeddings only from the configured folder, never a fallback copy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.v1_mnist.component.utils.som_data import resolve_embeddings_dir


def make_embeddings(folder: Path) -> Path:
    folder.mkdir(parents=True)
    np.save(folder / "train_features.npy", np.zeros((2, 3)))
    return folder


def test_relative_folder_is_taken_from_the_repo_root(tmp_path: Path) -> None:
    folder = make_embeddings(tmp_path / "outputs" / "embeddings")
    assert resolve_embeddings_dir("outputs/embeddings", tmp_path) == folder.resolve()


def test_absolute_folder_is_used_as_given(tmp_path: Path) -> None:
    folder = make_embeddings(tmp_path / "emb")
    assert resolve_embeddings_dir(folder, tmp_path / "elsewhere") == folder


def test_missing_folder_stops_even_if_another_copy_exists(tmp_path: Path) -> None:
    # A copy in a familiar place must not be picked up instead.
    make_embeddings(tmp_path / "outputs" / "old_copy")
    with pytest.raises(FileNotFoundError, match="paths.embeddings_dir"):
        resolve_embeddings_dir("outputs/new_run", tmp_path)


def test_folder_without_train_features_stops(tmp_path: Path) -> None:
    (tmp_path / "emb").mkdir()
    with pytest.raises(FileNotFoundError, match="train_features.npy"):
        resolve_embeddings_dir(tmp_path / "emb", tmp_path)
