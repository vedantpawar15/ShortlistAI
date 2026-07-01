"""Tests for weighted candidate scoring."""

from __future__ import annotations

import pytest
import pandas as pd

from src.scoring import CandidateScorer, ScoreWeights


# ---------------------------------------------------------------------------
# ScoreWeights validation
# ---------------------------------------------------------------------------


def test_default_weights_sum_to_one() -> None:
    weights = ScoreWeights()
    total = sum(weights.as_dict().values())
    assert abs(total - 1.0) < 1e-9


def test_weights_validate_passes_for_valid_weights() -> None:
    weights = ScoreWeights(
        semantic_similarity=0.30,
        skill_overlap=0.25,
        experience_match=0.15,
        education_match=0.05,
        title_similarity=0.10,
        location_match=0.05,
        behavioral_signal_score=0.10,
    )
    weights.validate()  # Should not raise.


def test_weights_validate_raises_for_negative_weight() -> None:
    weights = ScoreWeights(
        semantic_similarity=-0.10,
        skill_overlap=0.35,
        experience_match=0.25,
        education_match=0.15,
        title_similarity=0.10,
        location_match=0.10,
        behavioral_signal_score=0.15,
    )
    with pytest.raises(ValueError, match="non-negative"):
        weights.validate()


def test_weights_validate_raises_when_sum_not_one() -> None:
    weights = ScoreWeights(
        semantic_similarity=0.50,
        skill_overlap=0.50,
        experience_match=0.50,
        education_match=0.10,
        title_similarity=0.10,
        location_match=0.05,
        behavioral_signal_score=0.10,
    )
    with pytest.raises(ValueError, match="sum to 1"):
        weights.validate()


def test_weights_as_dict_returns_all_fields() -> None:
    weights = ScoreWeights()
    d = weights.as_dict()

    assert set(d.keys()) == {
        "semantic_similarity",
        "skill_overlap",
        "experience_match",
        "education_match",
        "title_similarity",
        "location_match",
        "behavioral_signal_score",
    }


# ---------------------------------------------------------------------------
# CandidateScorer.score()
# ---------------------------------------------------------------------------


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


def test_scorer_raises_for_missing_required_columns() -> None:
    features = pd.DataFrame([{"skill_overlap": 0.8, "experience_match": 0.7}])

    with pytest.raises(ValueError, match="missing required score columns"):
        CandidateScorer().score(features)


def test_scorer_returns_empty_series_for_empty_frame() -> None:
    result = CandidateScorer().score(pd.DataFrame())
    assert len(result) == 0


def test_scorer_clips_values_to_unit_range() -> None:
    # Pathological features > 1.0 should be clipped.
    features = pd.DataFrame(
        [{"semantic_similarity": 2.0, "skill_overlap": 5.0, "experience_match": 10.0, "education_match": 3.0}]
    )
    score = CandidateScorer().score(features).iloc[0]
    assert 0.0 <= score <= 1.0


def test_scorer_handles_nan_features_as_zero() -> None:
    import numpy as np

    features = pd.DataFrame(
        [
            {
                "semantic_similarity": float("nan"),
                "skill_overlap": 0.9,
                "experience_match": 0.8,
                "education_match": 0.7,
            }
        ]
    )
    score = CandidateScorer().score(features).iloc[0]
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize(
    "skill_overlap,expected_gt",
    [
        (1.0, 0.5),
        (0.0, 0.0),
    ],
)
def test_scorer_parametrized_skill_overlap_impact(skill_overlap: float, expected_gt: float) -> None:
    features = pd.DataFrame(
        [
            {
                "semantic_similarity": 0.5,
                "skill_overlap": skill_overlap,
                "experience_match": 0.5,
                "education_match": 0.5,
            }
        ]
    )
    score = CandidateScorer().score(features).iloc[0]
    assert score > expected_gt or (skill_overlap == 0.0 and score >= 0.0)


# ---------------------------------------------------------------------------
# CandidateScorer.score_breakdown()
# ---------------------------------------------------------------------------


def test_score_breakdown_returns_dataframe_with_same_length() -> None:
    features = pd.DataFrame(
        [
            {
                "candidate_id": "cand_a",
                "semantic_similarity": 0.8,
                "skill_overlap": 0.9,
                "experience_match": 0.7,
                "education_match": 0.6,
            },
            {
                "candidate_id": "cand_b",
                "semantic_similarity": 0.5,
                "skill_overlap": 0.4,
                "experience_match": 0.6,
                "education_match": 0.5,
            },
        ]
    )

    breakdown = CandidateScorer().score_breakdown(features)

    assert len(breakdown) == 2
    assert "semantic_similarity" in breakdown.columns
    assert "candidate_id" in breakdown.columns


def test_score_breakdown_contributions_sum_to_approximately_composite_score() -> None:
    features = pd.DataFrame(
        [
            {
                "candidate_id": "cand_a",
                "semantic_similarity": 0.8,
                "skill_overlap": 0.9,
                "experience_match": 0.7,
                "education_match": 0.6,
                "title_similarity": 0.5,
                "location_match": 1.0,
                "behavioral_signal_score": 0.8,
            }
        ]
    )
    scorer = CandidateScorer()
    composite = float(scorer.score(features).iloc[0])
    breakdown = scorer.score_breakdown(features)
    numeric_cols = [c for c in breakdown.columns if c != "candidate_id"]
    breakdown_sum = float(breakdown[numeric_cols].iloc[0].sum())

    assert abs(breakdown_sum - composite) < 0.01


def test_score_breakdown_returns_empty_for_empty_frame() -> None:
    breakdown = CandidateScorer().score_breakdown(pd.DataFrame())
    assert breakdown.empty
