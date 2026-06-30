"""Scoring primitives for candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for transparent candidate scoring."""

    semantic_similarity: float = 0.45
    skill_overlap: float = 0.35
    experience_match: float = 0.15
    education_match: float = 0.05


class CandidateScorer:
    """Compute transparent candidate-job fit scores."""

    def __init__(self, weights: ScoreWeights | None = None) -> None:
        self.weights = weights or ScoreWeights()

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Score candidates from engineered features."""
        raise NotImplementedError("Candidate scoring will be implemented later.")

