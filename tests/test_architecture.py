"""Architecture-level smoke tests."""

import numpy as np
import pandas as pd

from src.config import Settings
from src.ranking_engine import RankingEngine
from src.reranker import Reranker
from src.vector_store import RetrievalMatch


def test_settings_can_be_constructed() -> None:
    """Settings should be instantiable with defaults."""
    settings = Settings()
    assert settings.project_name == "RecruitAI"


def test_ranking_engine_can_be_constructed() -> None:
    """Ranking engine should expose the planned orchestration surface."""
    engine = RankingEngine()
    assert engine is not None


class FakeEmbeddingResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class FakeEmbeddingModel:
    def embed_candidates(self, candidates, use_cache=True, persist=True):
        return FakeEmbeddingResult(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    def embed_jobs(self, jobs, use_cache=True, persist=True):
        return FakeEmbeddingResult(np.asarray([[1.0, 0.0]], dtype=np.float32))


class FakeVectorStore:
    def build(self, embeddings, item_ids=None, metadata=None):
        self.item_ids = item_ids or []

    def retrieve(self, query_embedding, top_k=10):
        return [
            RetrievalMatch(item_id="cand_b", score=0.9, index=1, metadata={}),
            RetrievalMatch(item_id="cand_a", score=0.8, index=0, metadata={}),
        ][:top_k]


class FakeReranker(Reranker):
    def rerank(self, features, query_text=None, candidate_documents=None):
        return features.sort_values("candidate_id", ascending=False).reset_index(drop=True)


def test_ranking_engine_uses_retrieval_stage_when_dependencies_are_injected() -> None:
    candidates = pd.DataFrame(
        [
            {"candidate_id": "cand_a", "semantic_document": "python ml", "skills": "Python", "experience_years": 5},
            {"candidate_id": "cand_b", "semantic_document": "python nlp", "skills": "Python | NLP", "experience_years": 6},
        ]
    )
    job = pd.Series({"job_id": "job_1", "title": "ML Engineer", "description": "Python NLP", "experience_years": 5})
    engine = RankingEngine(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
        reranker=FakeReranker(),
        retrieval_top_k=2,
    )

    ranked = engine.rank_candidates(candidates, job)

    assert list(ranked["candidate_id"]) == ["cand_b", "cand_a"]

