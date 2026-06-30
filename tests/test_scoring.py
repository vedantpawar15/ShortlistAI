"""Tests for weighted candidate scoring."""

from __future__ import annotations

import pandas as pd

from src.scoring import CandidateScorer


def test_scorer_uses_optional_columns_when_present() -> None:
    features = pd.DataFrame(
        [
            {
                "semantic_similarity": 0.8,
                "skill_overlap": 0.9,
                "experience_match": 0.7,
                "education_match": 0.6,
                "title_similarity": 1.0,
                "location_match": 1.0,
                "behavioral_signal_score": 0.5,
            }
        ]
    )

    score = CandidateScorer().score(features).iloc[0]

    assert 0.0 <= score <= 1.0
    assert score > 0.75


def test_scorer_remains_backward_compatible_with_original_columns() -> None:
    features = pd.DataFrame(
        [
            {
                "semantic_similarity": 0.8,
                "skill_overlap": 0.9,
                "experience_match": 0.7,
                "education_match": 0.6,
            }
        ]
    )

    score = CandidateScorer().score(features).iloc[0]

    assert 0.0 <= score <= 1.0
