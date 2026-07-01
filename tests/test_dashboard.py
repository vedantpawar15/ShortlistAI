"""Tests for dashboard data helpers.

``load_dashboard_results`` is mocked to avoid reading real submission files.
``dashboard_metrics`` is tested directly as it is a pure function.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.dashboard import dashboard_metrics, load_dashboard_results


# ---------------------------------------------------------------------------
# dashboard_metrics — pure function, no mocking needed
# ---------------------------------------------------------------------------


def test_dashboard_metrics_summarize_ranked_results() -> None:
    metrics = dashboard_metrics(
        [
            {"candidate_id": "cand_a", "score": 0.8},
            {"candidate_id": "cand_b", "score": 0.6},
        ]
    )

    assert metrics["candidate_count"] == 2.0
    assert abs(metrics["average_score"] - 0.7) < 1e-9
    assert metrics["top_score"] == 0.8


def test_dashboard_metrics_handles_empty_input() -> None:
    metrics = dashboard_metrics([])

    assert metrics["candidate_count"] == 0.0
    assert metrics["average_score"] == 0.0
    assert metrics["top_score"] == 0.0


def test_dashboard_metrics_handles_single_candidate() -> None:
    metrics = dashboard_metrics([{"candidate_id": "cand_a", "score": 0.95}])

    assert metrics["candidate_count"] == 1.0
    assert metrics["average_score"] == 0.95
    assert metrics["top_score"] == 0.95


@pytest.mark.parametrize(
    "scores,expected_avg",
    [
        ([0.0, 1.0], 0.5),
        ([0.3, 0.6, 0.9], 0.6),
        ([0.5], 0.5),
    ],
)
def test_dashboard_metrics_average_score_parametrized(scores: list[float], expected_avg: float) -> None:
    rows = [{"candidate_id": f"c{i}", "score": s} for i, s in enumerate(scores)]
    metrics = dashboard_metrics(rows)

    assert abs(metrics["average_score"] - expected_avg) < 1e-9


# ---------------------------------------------------------------------------
# load_dashboard_results — mocked to avoid disk I/O
# ---------------------------------------------------------------------------


MOCK_RESULTS = [
    {"candidate_id": f"c{i}", "rank": i + 1, "score": round(0.9 - i * 0.1, 2), "reasoning": "OK"}
    for i in range(10)
]


@patch("src.dashboard.load_dashboard_results", return_value=MOCK_RESULTS[:5])
def test_load_dashboard_results_returns_requested_count(mock_load: object) -> None:
    results = load_dashboard_results(top_k=5)

    assert len(results) == 5
    assert {"candidate_id", "rank", "score", "reasoning"} <= set(results[0])


@patch("src.dashboard.load_dashboard_results", return_value=MOCK_RESULTS)
def test_load_dashboard_results_result_has_required_keys(mock_load: object) -> None:
    results = load_dashboard_results(top_k=10)

    for row in results:
        assert "candidate_id" in row
        assert "score" in row
