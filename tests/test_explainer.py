"""Tests for RankingExplainer — explanation generation and batch calibration."""

from __future__ import annotations

import pandas as pd
import pytest

from src.explainer import CandidateExplanation, RankingExplainer


EXPLAINER = RankingExplainer()


# ---------------------------------------------------------------------------
# explain() — single candidate
# ---------------------------------------------------------------------------


def test_explain_returns_candidate_explanation_model() -> None:
    features = {
        "score": 0.85,
        "semantic_similarity": 0.80,
        "lexical_similarity": 0.65,
        "skill_overlap": 0.90,
        "experience_match": 0.95,
        "education_match": 1.00,
        "title_similarity": 0.75,
        "location_match": 1.00,
        "behavioral_signal_score": 0.80,
        "matched_skills": ["python", "machine learning"],
        "missing_skills": [],
    }

    explanation = EXPLAINER.explain("cand_a", features)

    assert isinstance(explanation, CandidateExplanation)
    assert explanation.candidate_id == "cand_a"
    assert len(explanation.summary) > 0
    assert isinstance(explanation.strengths, list)
    assert isinstance(explanation.gaps, list)
    assert isinstance(explanation.score_breakdown, dict)


def test_explain_includes_all_available_features_in_breakdown() -> None:
    features = {
        "score": 0.75,
        "semantic_similarity": 0.70,
        "lexical_similarity": 0.60,
        "skill_overlap": 0.80,
        "experience_match": 0.90,
        "education_match": 0.80,
        "title_similarity": 0.50,
        "location_match": 0.50,
        "behavioral_signal_score": 0.60,
        "matched_skills": ["python"],
        "missing_skills": ["kubernetes"],
    }

    explanation = EXPLAINER.explain("cand_b", features)

    assert "semantic_similarity" in explanation.score_breakdown
    assert "skill_overlap" in explanation.score_breakdown
    assert "experience_match" in explanation.score_breakdown
    assert "education_match" in explanation.score_breakdown


def test_explain_generates_strengths_for_high_scoring_features() -> None:
    features = {
        "score": 0.90,
        "semantic_similarity": 0.90,
        "skill_overlap": 0.95,
        "experience_match": 0.95,
        "education_match": 1.00,
        "matched_skills": ["python", "machine learning", "nlp"],
        "missing_skills": [],
    }

    explanation = EXPLAINER.explain("cand_a", features, high_threshold=0.7)

    assert len(explanation.strengths) > 0
    assert any("skill" in s.lower() for s in explanation.strengths)


def test_explain_generates_gaps_for_low_scoring_features() -> None:
    features = {
        "score": 0.30,
        "semantic_similarity": 0.20,
        "skill_overlap": 0.10,
        "experience_match": 0.25,
        "education_match": 0.30,
        "matched_skills": [],
        "missing_skills": ["python", "machine learning", "kubernetes"],
    }

    explanation = EXPLAINER.explain("cand_z", features, low_threshold=0.4)

    assert len(explanation.gaps) > 0
    assert any("missing" in g.lower() for g in explanation.gaps)


def test_explain_with_percentile_rank_includes_context_in_summary() -> None:
    features = {"score": 0.85, "semantic_similarity": 0.80, "skill_overlap": 0.85, "experience_match": 1.0}

    explanation = EXPLAINER.explain("cand_a", features, percentile_rank=5.0)

    assert "top" in explanation.summary.lower()
    assert explanation.percentile_rank == 5.0


def test_explain_handles_missing_optional_features_gracefully() -> None:
    # Only the four required features — no title, location, behavioral.
    features = {
        "score": 0.60,
        "semantic_similarity": 0.60,
        "skill_overlap": 0.60,
        "experience_match": 0.60,
        "education_match": 0.60,
    }

    explanation = EXPLAINER.explain("cand_minimal", features)

    assert explanation.candidate_id == "cand_minimal"
    assert "semantic_similarity" in explanation.score_breakdown


def test_explain_handles_all_zero_features() -> None:
    features = {
        "score": 0.0,
        "semantic_similarity": 0.0,
        "skill_overlap": 0.0,
        "experience_match": 0.0,
        "education_match": 0.0,
        "matched_skills": [],
        "missing_skills": ["python"],
    }

    explanation = EXPLAINER.explain("cand_zero", features)

    assert explanation.candidate_id == "cand_zero"
    assert all(v == 0.0 for v in explanation.score_breakdown.values())


def test_explain_score_breakdown_values_are_rounded_to_4_decimals() -> None:
    features = {
        "score": 0.777777,
        "semantic_similarity": 0.333333,
        "skill_overlap": 0.666667,
        "experience_match": 0.5,
        "education_match": 1.0,
    }

    explanation = EXPLAINER.explain("cand_x", features)

    for value in explanation.score_breakdown.values():
        # Must not have more than 4 decimal places.
        assert round(value, 4) == value


# ---------------------------------------------------------------------------
# explain_batch() — batch calibration
# ---------------------------------------------------------------------------


def test_explain_batch_returns_one_explanation_per_candidate(ranked_df: pd.DataFrame) -> None:
    explanations = EXPLAINER.explain_batch(ranked_df)

    assert len(explanations) == len(ranked_df)


def test_explain_batch_sets_percentile_rank_correctly(ranked_df: pd.DataFrame) -> None:
    explanations = EXPLAINER.explain_batch(ranked_df)

    # First candidate (rank 1) should have the highest percentile.
    assert explanations[0].percentile_rank is not None
    assert explanations[0].percentile_rank >= explanations[-1].percentile_rank


def test_explain_batch_handles_empty_dataframe() -> None:
    explanations = EXPLAINER.explain_batch(pd.DataFrame())

    assert explanations == []


def test_explain_batch_candidate_ids_match_input_order(ranked_df: pd.DataFrame) -> None:
    explanations = EXPLAINER.explain_batch(ranked_df)

    for explanation, (_, row) in zip(explanations, ranked_df.iterrows()):
        assert explanation.candidate_id == str(row["candidate_id"])


@pytest.mark.parametrize("n_candidates", [1, 2, 5])
def test_explain_batch_works_for_various_pool_sizes(n_candidates: int) -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_id": f"c{i}",
                "score": round(0.9 - i * 0.1, 2),
                "semantic_similarity": 0.8,
                "skill_overlap": 0.7,
                "experience_match": 0.9,
                "education_match": 0.8,
            }
            for i in range(n_candidates)
        ]
    )

    explanations = EXPLAINER.explain_batch(df)

    assert len(explanations) == n_candidates
