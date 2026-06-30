"""Tests for candidate-job feature generation."""

from __future__ import annotations

import pandas as pd

from src.feature_engineering import FeatureEngineer, behavioral_score, location_score, title_match_score


def test_feature_engineer_adds_richer_matching_features() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "cand_1",
                "semantic_document": "python machine learning engineer",
                "skills": "Python | ML | NLP",
                "profile_current_title": "Senior ML Engineer",
                "location": "Pune",
                "redrob_signals_open_to_work_flag": True,
                "redrob_signals_github_activity_score": 80,
                "experience_years": 6,
                "education_0_degree": "B.Tech",
                "education_0_field_of_study": "Computer Science",
            }
        ]
    )
    job = pd.Series(
        {
            "title": "ML Engineer",
            "description": "Python NLP systems for production",
            "location": "Pune",
            "experience_years": 5,
        }
    )

    features = FeatureEngineer().build_features(candidates, job)
    row = features.iloc[0]

    assert row["title_similarity"] > 0.5
    assert row["location_match"] == 1.0
    assert row["behavioral_signal_score"] > 0.0
    assert row["matched_skill_count"] >= 2
    assert row["missing_skill_count"] >= 0


def test_title_and_location_scores_handle_missing_values() -> None:
    assert title_match_score("", "ML Engineer") == 0.0
    assert location_score({}, "Pune") == 0.0
    assert location_score({"location": "Pune"}, "") == 0.5


def test_behavioral_score_aggregates_numeric_and_boolean_signals() -> None:
    record = {
        "redrob_signals_open_to_work_flag": True,
        "redrob_signals_github_activity_score": 90,
        "redrob_signals_leadership_score": 70,
    }

    score = behavioral_score(record)

    assert 0.8 <= score <= 1.0
