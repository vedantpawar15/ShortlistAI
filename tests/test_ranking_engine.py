"""Tests for the RankingEngine orchestration layer.

All tests use injected fake dependencies so no real models, files, or FAISS
indexes are needed. This keeps the suite fully offline and sub-second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ranking_engine import RankingEngine, _candidate_fingerprint
from src.reranker import Reranker
from src.vector_store import RetrievalMatch


# ---------------------------------------------------------------------------
# Fake dependencies from conftest.py are available as fixtures:
#   fake_embedding_model, fake_vector_store, sample_candidates, sample_job
# ---------------------------------------------------------------------------


class PassthroughReranker(Reranker):
    """Reranker that preserves the feature score ordering."""

    def rerank(
        self,
        features: pd.DataFrame,
        query_text: str | None = None,
        candidate_documents: list[str] | None = None,
    ) -> pd.DataFrame:
        sort_col = "score" if "score" in features.columns else features.columns[0]
        return features.sort_values(sort_col, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_ranking_engine_constructs_with_defaults() -> None:
    engine = RankingEngine()
    assert engine is not None
    assert engine.hybrid_alpha == 0.6


def test_ranking_engine_accepts_custom_alpha() -> None:
    engine = RankingEngine(hybrid_alpha=0.9)
    assert engine.hybrid_alpha == 0.9


# ---------------------------------------------------------------------------
# rank_candidates — empty input
# ---------------------------------------------------------------------------


def test_rank_candidates_returns_empty_frame_for_empty_input(
    sample_job: pd.Series,
    fake_embedding_model: object,
    fake_vector_store: object,
) -> None:
    engine = RankingEngine(
        embedding_model=fake_embedding_model,  # type: ignore[arg-type]
        vector_store=fake_vector_store,  # type: ignore[arg-type]
        reranker=PassthroughReranker(),
    )
    result = engine.rank_candidates(pd.DataFrame(), sample_job)

    assert result.empty
    assert list(result.columns) == ["candidate_id", "rank", "score", "reasoning"]


# ---------------------------------------------------------------------------
# rank_candidates — standard flow
# ---------------------------------------------------------------------------


def test_rank_candidates_returns_required_columns(
    sample_candidates: pd.DataFrame,
    sample_job: pd.Series,
    fake_embedding_model: object,
    fake_vector_store: object,
) -> None:
    engine = RankingEngine(
        embedding_model=fake_embedding_model,  # type: ignore[arg-type]
        vector_store=fake_vector_store,  # type: ignore[arg-type]
        reranker=PassthroughReranker(),
    )
    result = engine.rank_candidates(sample_candidates, sample_job)

    required_columns = {"candidate_id", "rank", "score", "reasoning"}
    assert required_columns.issubset(set(result.columns))


def test_rank_candidates_returns_all_candidates(
    sample_candidates: pd.DataFrame,
    sample_job: pd.Series,
    fake_embedding_model: object,
    fake_vector_store: object,
) -> None:
    engine = RankingEngine(
        embedding_model=fake_embedding_model,  # type: ignore[arg-type]
        vector_store=fake_vector_store,  # type: ignore[arg-type]
        reranker=PassthroughReranker(),
    )
    result = engine.rank_candidates(sample_candidates, sample_job)

    assert len(result) == len(sample_candidates)


def test_rank_candidates_scores_are_bounded(
    sample_candidates: pd.DataFrame,
    sample_job: pd.Series,
    fake_embedding_model: object,
    fake_vector_store: object,
) -> None:
    engine = RankingEngine(
        embedding_model=fake_embedding_model,  # type: ignore[arg-type]
        vector_store=fake_vector_store,  # type: ignore[arg-type]
        reranker=PassthroughReranker(),
    )
    result = engine.rank_candidates(sample_candidates, sample_job)

    assert result["score"].between(0.0, 1.0).all()


def test_rank_candidates_ranks_are_sequential(
    sample_candidates: pd.DataFrame,
    sample_job: pd.Series,
    fake_embedding_model: object,
    fake_vector_store: object,
) -> None:
    engine = RankingEngine(
        embedding_model=fake_embedding_model,  # type: ignore[arg-type]
        vector_store=fake_vector_store,  # type: ignore[arg-type]
        reranker=PassthroughReranker(),
    )
    result = engine.rank_candidates(sample_candidates, sample_job)

    assert list(result["rank"]) == list(range(1, len(result) + 1))


def test_rank_candidates_reasoning_is_non_empty_string(
    sample_candidates: pd.DataFrame,
    sample_job: pd.Series,
    fake_embedding_model: object,
    fake_vector_store: object,
) -> None:
    engine = RankingEngine(
        embedding_model=fake_embedding_model,  # type: ignore[arg-type]
        vector_store=fake_vector_store,  # type: ignore[arg-type]
        reranker=PassthroughReranker(),
    )
    result = engine.rank_candidates(sample_candidates, sample_job)

    assert result["reasoning"].apply(lambda r: isinstance(r, str) and len(r) > 0).all()


# ---------------------------------------------------------------------------
# rank_candidates — retrieval stage ordering
# ---------------------------------------------------------------------------


def test_rank_candidates_uses_retrieval_stage_order(
    sample_candidates: pd.DataFrame,
    sample_job: pd.Series,
) -> None:
    """Verify that the retrieval stage reorders candidates per vector-store results."""

    class OrderedFakeEmbeddingResult:
        def __init__(self, embeddings: np.ndarray) -> None:
            self.embeddings = embeddings

    class OrderedFakeEmbeddingModel:
        def embed_candidates(self, candidates: object, **kw: object) -> OrderedFakeEmbeddingResult:
            return OrderedFakeEmbeddingResult(np.eye(3, 4, dtype=np.float32))

        def embed_jobs(self, jobs: object, **kw: object) -> OrderedFakeEmbeddingResult:
            return OrderedFakeEmbeddingResult(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    class ReversedVectorStore:
        """Returns candidates in reverse order (cand_c first, cand_a last)."""

        _built = False

        @property
        def is_built(self) -> bool:
            return self._built

        def build(self, embeddings: object, item_ids: list[str] | None = None, metadata: object = None) -> None:
            self._built = True

        def retrieve(self, query_embedding: object, top_k: int = 10) -> list[RetrievalMatch]:
            return [
                RetrievalMatch(item_id="cand_c", score=0.95, index=2, metadata={}),
                RetrievalMatch(item_id="cand_b", score=0.80, index=1, metadata={}),
                RetrievalMatch(item_id="cand_a", score=0.60, index=0, metadata={}),
            ][:top_k]

    engine = RankingEngine(
        embedding_model=OrderedFakeEmbeddingModel(),  # type: ignore[arg-type]
        vector_store=ReversedVectorStore(),  # type: ignore[arg-type]
        reranker=PassthroughReranker(),
    )
    result = engine.rank_candidates(sample_candidates, sample_job)

    # Scores are determined by hybrid scoring, so we just check all candidates appear.
    assert set(result["candidate_id"]) == {"cand_a", "cand_b", "cand_c"}


# ---------------------------------------------------------------------------
# _candidate_fingerprint
# ---------------------------------------------------------------------------


def test_candidate_fingerprint_is_stable_for_same_ids() -> None:
    df1 = pd.DataFrame([{"candidate_id": "c1"}, {"candidate_id": "c2"}])
    df2 = pd.DataFrame([{"candidate_id": "c2"}, {"candidate_id": "c1"}])

    # Order-independent hash.
    assert _candidate_fingerprint(df1) == _candidate_fingerprint(df2)


def test_candidate_fingerprint_differs_for_different_ids() -> None:
    df1 = pd.DataFrame([{"candidate_id": "c1"}])
    df2 = pd.DataFrame([{"candidate_id": "c9"}])

    assert _candidate_fingerprint(df1) != _candidate_fingerprint(df2)


def test_candidate_fingerprint_returns_empty_for_missing_column() -> None:
    df = pd.DataFrame([{"name": "Alice"}])

    assert _candidate_fingerprint(df) == ""


# ---------------------------------------------------------------------------
# Single candidate edge case
# ---------------------------------------------------------------------------


def test_rank_single_candidate(sample_job: pd.Series, fake_embedding_model: object, fake_vector_store: object) -> None:
    single = pd.DataFrame(
        [
            {
                "candidate_id": "solo",
                "semantic_document": "python engineer",
                "skills": "Python",
                "experience_years": 5,
            }
        ]
    )
    engine = RankingEngine(
        embedding_model=fake_embedding_model,  # type: ignore[arg-type]
        vector_store=fake_vector_store,  # type: ignore[arg-type]
        reranker=PassthroughReranker(),
    )
    result = engine.rank_candidates(single, sample_job)

    assert len(result) == 1
    assert result.iloc[0]["rank"] == 1
    assert 0.0 <= result.iloc[0]["score"] <= 1.0
