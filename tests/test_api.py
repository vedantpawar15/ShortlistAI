"""Tests for the local API surface."""

from __future__ import annotations

from src.api import RankRequest, dataset_summary, health, rank_candidates


def test_health_reports_project_name() -> None:
    response = health()

    assert response.status == "ok"
    assert response.project == "RecruitAI"


def test_dataset_summary_lists_local_job_ids() -> None:
    response = dataset_summary()

    assert response.candidates == 50
    assert "job_description" in response.jobs


def test_rank_candidates_uses_requested_job_id() -> None:
    response = rank_candidates(RankRequest(job_id="job_description", top_k=5))

    assert response.job_id == "job_description"
    assert len(response.results) == 5
