"""Tests for dashboard data helpers."""

from __future__ import annotations

from src.dashboard import dashboard_metrics, load_dashboard_results


def test_dashboard_metrics_summarize_ranked_results() -> None:
    metrics = dashboard_metrics(
        [
            {"candidate_id": "cand_a", "score": 0.8},
            {"candidate_id": "cand_b", "score": 0.6},
        ]
    )

    assert metrics["candidate_count"] == 2.0
    assert metrics["average_score"] == 0.7
    assert metrics["top_score"] == 0.8


def test_load_dashboard_results_reads_submission_file() -> None:
    results = load_dashboard_results(top_k=5)

    assert len(results) == 5
    assert {"candidate_id", "rank", "score", "reasoning"} <= set(results[0])
