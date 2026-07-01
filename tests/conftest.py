"""Shared pytest fixtures for the ShortlistAI test suite.

All fixtures that need disk access use ``tmp_path`` (pytest-managed) rather than
``tempfile.mkdtemp()`` so that pytest controls cleanup.

Integration-heavy components (DataLoader, RankingEngine) are provided as
lightweight fakes so that unit tests remain fast and offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.vector_store import RetrievalMatch


# ---------------------------------------------------------------------------
# Reusable data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_candidates() -> pd.DataFrame:
    """Minimal candidate DataFrame sufficient for unit tests."""
    return pd.DataFrame(
        [
            {
                "candidate_id": "cand_a",
                "semantic_document": "python machine learning engineer five years experience",
                "skills": "Python | ML | NLP",
                "profile_current_title": "Senior ML Engineer",
                "location": "Pune",
                "experience_years": 5,
                "education_0_degree": "B.Tech",
                "education_0_field_of_study": "Computer Science",
                "redrob_signals_open_to_work_flag": True,
                "redrob_signals_github_activity_score": 80,
            },
            {
                "candidate_id": "cand_b",
                "semantic_document": "nlp natural language processing deep learning python",
                "skills": "Python | NLP | PyTorch",
                "profile_current_title": "NLP Engineer",
                "location": "Bangalore",
                "experience_years": 7,
                "education_0_degree": "M.S.",
                "education_0_field_of_study": "AI",
                "redrob_signals_open_to_work_flag": False,
                "redrob_signals_github_activity_score": 92,
            },
            {
                "candidate_id": "cand_c",
                "semantic_document": "data scientist sql tableau visualization",
                "skills": "SQL | Tableau | Python",
                "profile_current_title": "Data Scientist",
                "location": "Mumbai",
                "experience_years": 3,
                "education_0_degree": "B.S.",
                "education_0_field_of_study": "Statistics",
                "redrob_signals_github_activity_score": 45,
            },
        ]
    )


@pytest.fixture()
def sample_job() -> pd.Series:
    """Minimal job Series for unit tests."""
    return pd.Series(
        {
            "job_id": "job_1",
            "title": "ML Engineer",
            "description": "Build production ML systems with Python and NLP. 5+ years required.",
            "required_skills": ["Python", "ML", "NLP"],
            "location": "Pune",
            "experience_years": 5,
        }
    )


@pytest.fixture()
def ranked_df() -> pd.DataFrame:
    """Pre-built ranked DataFrame for explainer and dashboard tests."""
    return pd.DataFrame(
        [
            {
                "candidate_id": "cand_a",
                "rank": 1,
                "score": 0.85,
                "semantic_similarity": 0.82,
                "lexical_similarity": 0.65,
                "skill_overlap": 0.90,
                "experience_match": 1.00,
                "education_match": 1.00,
                "title_similarity": 0.72,
                "location_match": 1.00,
                "behavioral_signal_score": 0.80,
                "matched_skills": ["python", "machine learning"],
                "missing_skills": [],
                "reasoning": "Strong semantic alignment.",
            },
            {
                "candidate_id": "cand_b",
                "rank": 2,
                "score": 0.72,
                "semantic_similarity": 0.70,
                "lexical_similarity": 0.55,
                "skill_overlap": 0.70,
                "experience_match": 1.00,
                "education_match": 1.00,
                "title_similarity": 0.50,
                "location_match": 0.20,
                "behavioral_signal_score": 0.92,
                "matched_skills": ["python", "natural language processing"],
                "missing_skills": ["machine learning"],
                "reasoning": "Good NLP fit.",
            },
            {
                "candidate_id": "cand_c",
                "rank": 3,
                "score": 0.45,
                "semantic_similarity": 0.30,
                "lexical_similarity": 0.30,
                "skill_overlap": 0.33,
                "experience_match": 0.60,
                "education_match": 0.50,
                "title_similarity": 0.20,
                "location_match": 0.20,
                "behavioral_signal_score": 0.45,
                "matched_skills": ["python"],
                "missing_skills": ["machine learning", "natural language processing"],
                "reasoning": "Below threshold on most signals.",
            },
        ]
    )


# ---------------------------------------------------------------------------
# Fake ML components (no disk I/O, no network)
# ---------------------------------------------------------------------------


class FakeEmbeddingResult:
    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings


class FakeEmbeddingModel:
    """Deterministic embedding model for unit tests — no sentence-transformers."""

    def embed_candidates(
        self,
        candidates: Any,
        use_cache: bool = True,
        persist: bool = True,
    ) -> FakeEmbeddingResult:
        n = len(candidates) if hasattr(candidates, "__len__") else 2
        return FakeEmbeddingResult(np.eye(n, 4, dtype=np.float32))

    def embed_jobs(
        self,
        jobs: Any,
        use_cache: bool = True,
        persist: bool = True,
    ) -> FakeEmbeddingResult:
        return FakeEmbeddingResult(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))


class FakeVectorStore:
    """In-memory vector store that always returns the first two item IDs."""

    def __init__(self) -> None:
        self.item_ids: list[str] = []
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built

    def build(
        self,
        embeddings: np.ndarray,
        item_ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        self.item_ids = item_ids or []
        self._built = True

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 10) -> list[RetrievalMatch]:
        return [
            RetrievalMatch(item_id=cid, score=0.9 - i * 0.1, index=i, metadata={})
            for i, cid in enumerate(self.item_ids[:top_k])
        ]


@pytest.fixture()
def fake_embedding_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture()
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


# ---------------------------------------------------------------------------
# Data directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_dir_with_candidates(tmp_path: Path) -> Path:
    """Temporary data directory with a minimal candidates JSON file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    candidates = [
        {"candidate_id": "c1", "name": "Alice", "skills": "Python | ML", "experience_years": 5},
        {"candidate_id": "c2", "name": "Bob", "skills": "Java | SQL", "experience_years": 3},
    ]
    (data_dir / "sample_candidates.json").write_text(
        json.dumps(candidates), encoding="utf-8"
    )
    return data_dir


@pytest.fixture()
def data_dir_with_jobs(tmp_path: Path) -> Path:
    """Temporary data directory with a minimal jobs JSONL file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    jobs = [
        {
            "job_id": "j1",
            "title": "ML Engineer",
            "description": "Python ML systems",
            "required_skills": ["Python", "ML"],
            "experience_years": 5,
        }
    ]
    (data_dir / "jobs.jsonl").write_text(
        "\n".join(json.dumps(j) for j in jobs), encoding="utf-8"
    )
    return data_dir
