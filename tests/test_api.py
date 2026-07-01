"""Tests for the local API surface.

All tests that previously called the real ML pipeline now use unittest.mock
to substitute DataLoader and RankingEngine, keeping the suite fast and offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.api import RankRequest, dataset_summary, health, rank_candidates


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_reports_project_name() -> None:
    response = health()

    assert response.status == "ok"
    assert response.project == "RecruitAI"


# ---------------------------------------------------------------------------
# Dataset summary — mocked DataLoader
# ---------------------------------------------------------------------------


def _make_mock_loader(
    n_candidates: int = 5,
    job_ids: list[str] | None = None,
) -> MagicMock:
    mock_loader = MagicMock()
    mock_loader.load_candidates.return_value = pd.DataFrame(
        [{"candidate_id": f"c{i}"} for i in range(n_candidates)]
    )
    mock_loader.load_jobs.return_value = pd.DataFrame(
        [{"job_id": jid, "title": "ML Engineer"} for jid in (job_ids or ["job_description"])]
    )
    return mock_loader


@patch("src.api.DataLoader")
def test_dataset_summary_returns_candidate_count_and_job_ids(mock_loader_cls: MagicMock) -> None:
    mock_loader_cls.return_value = _make_mock_loader(n_candidates=42, job_ids=["job_description"])

    response = dataset_summary()

    assert response.candidates == 42
    assert "job_description" in response.jobs


@patch("src.api.DataLoader")
def test_dataset_summary_handles_multiple_jobs(mock_loader_cls: MagicMock) -> None:
    mock_loader_cls.return_value = _make_mock_loader(
        n_candidates=10,
        job_ids=["job_a", "job_b"],
    )

    response = dataset_summary()

    assert response.candidates == 10
    assert set(response.jobs) == {"job_a", "job_b"}


# ---------------------------------------------------------------------------
# Rank candidates — mocked DataLoader + RankingEngine
# ---------------------------------------------------------------------------


def _make_ranked_df(top_k: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": f"c{i}",
                "rank": i + 1,
                "score": round(0.9 - i * 0.05, 2),
                "reasoning": f"Candidate {i} explanation.",
            }
            for i in range(top_k)
        ]
    )


@patch("src.api.RankingEngine")
@patch("src.api.DataLoader")
def test_rank_candidates_returns_requested_job_id_and_top_k(
    mock_loader_cls: MagicMock,
    mock_engine_cls: MagicMock,
) -> None:
    mock_loader = _make_mock_loader(n_candidates=20, job_ids=["job_description"])
    mock_loader_cls.return_value = mock_loader
    mock_engine = MagicMock()
    mock_engine.rank_candidates.return_value = _make_ranked_df(top_k=10)
    mock_engine_cls.return_value = mock_engine

    response = rank_candidates(RankRequest(job_id="job_description", top_k=5))

    assert response.job_id == "job_description"
    assert len(response.results) == 5


@patch("src.api.RankingEngine")
@patch("src.api.DataLoader")
def test_rank_candidates_raises_value_error_for_unknown_job(
    mock_loader_cls: MagicMock,
    mock_engine_cls: MagicMock,
) -> None:
    mock_loader = _make_mock_loader(n_candidates=5, job_ids=["job_description"])
    mock_loader_cls.return_value = mock_loader

    with pytest.raises(ValueError, match="Job 'nonexistent_job' was not found"):
        rank_candidates(RankRequest(job_id="nonexistent_job", top_k=5))


@patch("src.api.RankingEngine")
@patch("src.api.DataLoader")
def test_rank_candidates_respects_top_k_limit(
    mock_loader_cls: MagicMock,
    mock_engine_cls: MagicMock,
) -> None:
    mock_loader = _make_mock_loader(n_candidates=50, job_ids=["job_description"])
    mock_loader_cls.return_value = mock_loader
    mock_engine = MagicMock()
    mock_engine.rank_candidates.return_value = _make_ranked_df(top_k=50)
    mock_engine_cls.return_value = mock_engine

    response = rank_candidates(RankRequest(job_id="job_description", top_k=3))

    assert len(response.results) == 3


@pytest.mark.parametrize("top_k", [1, 5, 10])
@patch("src.api.RankingEngine")
@patch("src.api.DataLoader")
def test_rank_candidates_parametrized_top_k(
    mock_loader_cls: MagicMock,
    mock_engine_cls: MagicMock,
    top_k: int,
) -> None:
    mock_loader = _make_mock_loader(n_candidates=20, job_ids=["job_description"])
    mock_loader_cls.return_value = mock_loader
    mock_engine = MagicMock()
    mock_engine.rank_candidates.return_value = _make_ranked_df(top_k=20)
    mock_engine_cls.return_value = mock_engine

    response = rank_candidates(RankRequest(job_id="job_description", top_k=top_k))

    assert len(response.results) == top_k
