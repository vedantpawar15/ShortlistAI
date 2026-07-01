"""Scoring primitives for candidate ranking.

Design notes
------------
* ``ScoreWeights`` is a frozen dataclass so weights are immutable after construction.
* ``validate()`` asserts that all weights are non-negative and sum to 1.0 (±ε).
* ``CandidateScorer.score()`` performs the weighted sum and clips to [0, 1].
* ``CandidateScorer.score_breakdown()`` returns a per-candidate DataFrame with
  individual weighted contributions, useful for explainability and debugging.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for transparent candidate scoring.

    ``semantic_similarity`` is the embedding cosine score when the FAISS
    retrieval stage is active; it falls back to the best lexical signal
    otherwise.  The remaining weights cover interpretable feature dimensions.

    The weights must be non-negative and sum to 1.0.
    """

    semantic_similarity: float = 0.30
    skill_overlap: float = 0.25
    experience_match: float = 0.15
    education_match: float = 0.05
    title_similarity: float = 0.10
    location_match: float = 0.05
    behavioral_signal_score: float = 0.10

    def validate(self) -> None:
        """Raise ValueError if any weight is negative or the sum deviates from 1."""
        weight_values = astuple(self)
        for field, value in zip(fields(self), weight_values):
            if value < 0:
                raise ValueError(f"Weight '{field.name}' must be non-negative, got {value}")
        total = sum(weight_values)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")

    def as_dict(self) -> dict[str, float]:
        """Return weights as a plain dictionary."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


class CandidateScorer:
    """Compute transparent candidate-job fit scores."""

    def __init__(self, weights: ScoreWeights | None = None) -> None:
        self.weights = weights or ScoreWeights()
        self.weights.validate()

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Score candidates from engineered features.

        Required columns: ``semantic_similarity``, ``skill_overlap``,
        ``experience_match``, ``education_match``.

        Optional columns (included when present): ``title_similarity``,
        ``location_match``, ``behavioral_signal_score``.

        The final score is the weighted sum renormalised by the sum of active
        weights so that optional-column absence does not deflate scores.
        """
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

        weighted = sum(
            features[column].fillna(0.0).astype(float) * weight
            for column, weight in active_columns.items()
        )
        total_weight = sum(active_columns.values())
        return (weighted / total_weight).clip(lower=0.0, upper=1.0)

    def score_breakdown(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return per-feature weighted score contributions for each candidate.

        Each column in the result is the *weighted contribution* of that feature
        to the final score (i.e. ``feature_value × weight / total_weight``).
        This is the primary input for explainer visualisations.
        """
        if features.empty:
            return pd.DataFrame()

        all_columns = {**{
            "semantic_similarity": self.weights.semantic_similarity,
            "skill_overlap": self.weights.skill_overlap,
            "experience_match": self.weights.experience_match,
            "education_match": self.weights.education_match,
            "title_similarity": self.weights.title_similarity,
            "location_match": self.weights.location_match,
            "behavioral_signal_score": self.weights.behavioral_signal_score,
        }}

        active_columns = {col: w for col, w in all_columns.items() if col in features}
        total_weight = sum(active_columns.values())

        breakdown: dict[str, pd.Series] = {}
        for column, weight in active_columns.items():
            breakdown[column] = (
                features[column].fillna(0.0).astype(float) * weight / total_weight
            ).clip(lower=0.0, upper=1.0)

        result = pd.DataFrame(breakdown)
        if "candidate_id" in features.columns:
            result.insert(0, "candidate_id", features["candidate_id"].values)
        return result
