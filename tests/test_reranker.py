"""Tests for reranking behavior."""

from __future__ import annotations

import pandas as pd

from src.reranker import PRIMARY_RERANKER_MODEL_NAME, Reranker, lexical_rerank_score


class FakeCrossEncoder:
    """Deterministic cross-encoder stub for reranker tests."""

    def predict(self, pairs):
        scores = []
        for query, document in pairs:
            if "python" in document.lower():
                scores.append(0.9)
            else:
                scores.append(0.2)
        return scores


def test_default_reranker_uses_bge_model_name() -> None:
    reranker = Reranker()

    assert reranker.model_name == PRIMARY_RERANKER_MODEL_NAME


def test_reranker_uses_cross_encoder_scores_when_query_and_documents_are_available() -> None:
    reranker = Reranker(model=FakeCrossEncoder())
    features = pd.DataFrame(
        [
            {"candidate_id": "cand_a", "score": 0.4},
            {"candidate_id": "cand_b", "score": 0.8},
        ]
    )

    ranked = reranker.rerank(
        features,
        query_text="Senior ML engineer",
        candidate_documents=["Python systems experience", "Operations background"],
    )

    # After normalize_scores(), [0.9, 0.2] → [1.0, 0.0].
    # cand_a (Python doc, score=1.0) should rank above cand_b (Operations, score=0.0).
    assert list(ranked["candidate_id"]) == ["cand_a", "cand_b"]
    assert ranked.iloc[0]["rerank_score"] == 1.0
    assert ranked.iloc[1]["rerank_score"] == 0.0


def test_reranker_falls_back_to_existing_score_sort_without_model_inputs() -> None:
    reranker = Reranker()
    features = pd.DataFrame(
        [
            {"candidate_id": "cand_a", "score": 0.4, "semantic_similarity": 0.1},
            {"candidate_id": "cand_b", "score": 0.8, "semantic_similarity": 0.2},
        ]
    )

    ranked = reranker.rerank(features)

    assert list(ranked["candidate_id"]) == ["cand_b", "cand_a"]


def test_lexical_rerank_score_rewards_more_similar_documents() -> None:
    higher = lexical_rerank_score("python machine learning", "python machine learning engineer")
    lower = lexical_rerank_score("python machine learning", "enterprise sales leader")

    assert higher > lower
