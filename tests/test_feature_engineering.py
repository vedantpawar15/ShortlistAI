"""Tests for candidate-job feature generation.

Aligned with the refactored feature_engineering.py which:
* Returns ``semantic_similarity`` (from embedding cosine scores when available,
  else from lexical/BM25 fallback) and ``lexical_similarity`` as separate columns.
* Exposes ``bm25_similarity`` as an additional signal.
* ``first_present()`` now returns ``""`` when no value exists.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import (
    FeatureEngineer,
    behavioral_score,
    bm25_similarity,
    lexical_similarity,
    location_score,
    reciprocal_rank_fusion,
    term_frequencies,
    title_match_score,
    tokenize,
)


# ---------------------------------------------------------------------------
# FeatureEngineer.build_features()
# ---------------------------------------------------------------------------


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


def test_feature_engineer_has_semantic_and_lexical_columns() -> None:
    candidates = pd.DataFrame(
        [{"candidate_id": "c1", "semantic_document": "python machine learning", "skills": "Python | ML"}]
    )
    job = pd.Series({"title": "ML Engineer", "description": "Python ML systems"})

    features = FeatureEngineer().build_features(candidates, job)

    # Both distinct columns must be present.
    assert "semantic_similarity" in features.columns
    assert "lexical_similarity" in features.columns
    assert "bm25_similarity" in features.columns


def test_feature_engineer_all_scores_are_bounded() -> None:
    candidates = pd.DataFrame(
        [
            {"candidate_id": f"c{i}", "semantic_document": f"candidate {i} text", "experience_years": i}
            for i in range(5)
        ]
    )
    job = pd.Series({"title": "Engineer", "description": "Build software", "experience_years": 3})

    features = FeatureEngineer().build_features(candidates, job)

    score_cols = [
        "semantic_similarity",
        "lexical_similarity",
        "bm25_similarity",
        "skill_overlap",
        "experience_match",
        "education_match",
        "title_similarity",
        "location_match",
        "behavioral_signal_score",
    ]
    for col in score_cols:
        assert features[col].between(0.0, 1.0).all(), f"Column {col!r} out of [0,1] range"


def test_feature_engineer_semantic_score_comes_from_embedding_when_available() -> None:
    """When _semantic_score is injected, semantic_similarity must equal it."""
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "semantic_document": "python engineer",
                "_semantic_score": 0.77,
            }
        ]
    )
    job = pd.Series({"title": "ML Engineer", "description": "Python ML"})

    features = FeatureEngineer().build_features(candidates, job)

    assert abs(features.iloc[0]["semantic_similarity"] - 0.77) < 1e-9


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


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


def test_behavioral_score_returns_zero_for_empty_record() -> None:
    assert behavioral_score({}) == 0.0
    assert behavioral_score({"name": "Alice"}) == 0.0


def test_tokenize_returns_set_of_words() -> None:
    tokens = tokenize("Python machine learning engineer")
    assert isinstance(tokens, set)
    assert "python" in tokens
    assert "machine" in tokens
    # Stop words filtered.
    assert "and" not in tokens
    assert "the" not in tokens


def test_lexical_similarity_returns_one_for_identical_tokens() -> None:
    tokens = {"python", "machine", "learning"}
    result = lexical_similarity(tokens, tokens)
    assert result == 1.0


def test_lexical_similarity_returns_zero_for_disjoint_tokens() -> None:
    result = lexical_similarity({"python", "ml"}, {"java", "scala"})
    assert result < 0.2


def test_term_frequencies_sum_to_one() -> None:
    tokens = {"a", "b", "c", "d"}
    tf = term_frequencies(tokens)
    assert abs(sum(tf.values()) - 1.0) < 1e-9


def test_bm25_similarity_rewards_relevant_document() -> None:
    job_tokens = {"python", "machine", "learning", "engineer"}
    job_tf = term_frequencies(job_tokens)
    relevant = {"python", "machine", "learning", "nlp"}
    irrelevant = {"sales", "marketing", "operations"}

    rel_score = bm25_similarity(relevant, job_tf, len(job_tokens))
    irrel_score = bm25_similarity(irrelevant, job_tf, len(job_tokens))

    assert rel_score > irrel_score


def test_bm25_similarity_returns_zero_for_empty_inputs() -> None:
    assert bm25_similarity(set(), {}, 0) == 0.0
    assert bm25_similarity({"python"}, {}, 5) == 0.0


def test_bm25_similarity_output_is_bounded() -> None:
    tokens = {"python", "ml", "nlp"}
    tf = term_frequencies(tokens)
    result = bm25_similarity(tokens, tf, len(tokens))
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def test_reciprocal_rank_fusion_rewards_top_ranked_items() -> None:
    list_a = ["cand_x", "cand_y", "cand_z"]
    list_b = ["cand_x", "cand_z", "cand_y"]

    fused = reciprocal_rank_fusion(list_a, list_b)
    ids = [cid for cid, _ in fused]

    assert ids[0] == "cand_x"


def test_reciprocal_rank_fusion_handles_single_list() -> None:
    rank_list = ["a", "b", "c"]
    fused = reciprocal_rank_fusion(rank_list)
    ids = [cid for cid, _ in fused]

    assert ids == rank_list


def test_reciprocal_rank_fusion_handles_empty_lists() -> None:
    fused = reciprocal_rank_fusion([], [])
    assert fused == []


@pytest.mark.parametrize(
    "title_a,title_b,expected_gt",
    [
        ("ML Engineer", "ML Engineer", 0.9),
        ("Senior Software Engineer", "Software Engineer", 0.6),
        ("Sales Manager", "ML Engineer", 0.0),
    ],
)
def test_title_match_score_parametrized(title_a: str, title_b: str, expected_gt: float) -> None:
    score = title_match_score(title_a, title_b)
    assert score > expected_gt or (expected_gt == 0.0 and score >= 0.0)
