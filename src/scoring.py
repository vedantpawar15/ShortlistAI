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
        if features.empty:
            return pd.Series(dtype=float)
        required_columns = {
            "semantic_similarity": self.weights.semantic_similarity,
            "skill_overlap": self.weights.skill_overlap,
            "experience_match": self.weights.experience_match,
            "education_match": self.weights.education_match,
        }
        missing = [column for column in required_columns if column not in features]
        if missing:
            raise ValueError(f"Feature frame is missing required score columns: {missing}")

        weighted = sum(features[column].fillna(0.0).astype(float) * weight for column, weight in required_columns.items())
        total_weight = sum(required_columns.values())
        return (weighted / total_weight).clip(lower=0.0, upper=1.0)

