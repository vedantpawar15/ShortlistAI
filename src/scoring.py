"""Scoring primitives for candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for transparent candidate scoring."""

    semantic_similarity: float = 0.30
    skill_overlap: float = 0.25
    experience_match: float = 0.15
    education_match: float = 0.05
    title_similarity: float = 0.10
    location_match: float = 0.05
    behavioral_signal_score: float = 0.10


class CandidateScorer:
    """Compute transparent candidate-job fit scores."""

    def __init__(self, weights: ScoreWeights | None = None) -> None:
        self.weights = weights or ScoreWeights()

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Score candidates from engineered features."""
        if features.empty:
            return pd.Series(dtype=float)
        base_columns = {
            "semantic_similarity": self.weights.semantic_similarity,
            "skill_overlap": self.weights.skill_overlap,
            "experience_match": self.weights.experience_match,
            "education_match": self.weights.education_match,
        }
        optional_columns = {
            "title_similarity": self.weights.title_similarity,
            "location_match": self.weights.location_match,
            "behavioral_signal_score": self.weights.behavioral_signal_score,
        }
        missing = [column for column in base_columns if column not in features]
        if missing:
            raise ValueError(f"Feature frame is missing required score columns: {missing}")

        active_columns = dict(base_columns)
        active_columns.update({column: weight for column, weight in optional_columns.items() if column in features})
        weighted = sum(features[column].fillna(0.0).astype(float) * weight for column, weight in active_columns.items())
        total_weight = sum(active_columns.values())
        return (weighted / total_weight).clip(lower=0.0, upper=1.0)

