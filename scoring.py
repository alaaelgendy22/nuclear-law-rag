"""Dependency-light score normalization and hybrid fusion."""

from __future__ import annotations

import numpy as np


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return scores
    min_s = scores.min()
    max_s = scores.max()
    if max_s == min_s:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def fuse_hybrid_scores(bm25_scores: np.ndarray, embedding_scores: np.ndarray,
                       alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if len(bm25_scores) != len(embedding_scores):
        raise ValueError("score arrays must have the same length")
    return alpha * min_max_normalize(bm25_scores) + (1 - alpha) * min_max_normalize(embedding_scores)
