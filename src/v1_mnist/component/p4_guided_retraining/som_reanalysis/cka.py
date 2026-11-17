"""Linear CKA (Kornblith et al., 2019): how similar are two embeddings of the same images?"""

from __future__ import annotations

import numpy as np


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered kernel alignment between two (images x features) matrices, in 0..1."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("Both embeddings need the same images in the same order.")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, "fro") ** 2
    denom = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    if denom == 0:
        raise ValueError("An embedding has no variance.")
    return float(cross / denom)


def cka_matrix(embeddings: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Pairwise linear CKA of all named embeddings."""
    names = list(embeddings)
    mat = np.eye(len(names))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            mat[i, j] = mat[j, i] = linear_cka(embeddings[a], embeddings[names[j]])
    return names, mat


def group_means(names: list[str], mat: np.ndarray, group_of: dict[str, str]) -> dict[str, float]:
    """Mean CKA within each group (for example arm) and between different groups."""
    within: dict[str, list[float]] = {}
    between: list[float] = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            if group_of[a] == group_of[names[j]]:
                within.setdefault(group_of[a], []).append(mat[i, j])
            else:
                between.append(mat[i, j])
    out = {f"within_{g}": float(np.mean(v)) for g, v in within.items()}
    if between:
        out["between_groups"] = float(np.mean(between))
    return out
