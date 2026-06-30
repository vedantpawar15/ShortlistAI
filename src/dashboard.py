"""Streamlit dashboard for recruiter workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import plotly.express as px
except ImportError:
    px = None

try:
    import streamlit as st
except ImportError:
    st = None

from src.config import get_settings
from src.data_loader import DataLoader
from src.ranking_engine import RankingEngine


def load_dashboard_results(top_k: int = 100) -> list[dict[str, object]]:
    """Load existing results or generate ranked candidates for the dashboard."""
    submission_path = Path("submission.csv")
    if submission_path.exists():
        ranked_frame = pd.read_csv(submission_path).head(top_k)
        return ranked_frame.to_dict(orient="records")

    settings = get_settings()
    loader = DataLoader(settings.data_dir)
    candidates = loader.load_candidates("sample_candidates.json")
    jobs = loader.load_jobs("job_description.docx")
    ranked_frame = RankingEngine().rank_candidates(candidates, jobs.iloc[0]).head(top_k)
    return ranked_frame.to_dict(orient="records")


def dashboard_metrics(ranked: list[dict[str, object]]) -> dict[str, float]:
    """Compute high-level dashboard metrics from ranked results."""
    if not ranked:
        return {"candidate_count": 0.0, "average_score": 0.0, "top_score": 0.0}
    scores = [float(row.get("score", 0.0) or 0.0) for row in ranked]
    return {
        "candidate_count": float(len(ranked)),
        "average_score": float(sum(scores) / len(scores)),
        "top_score": float(max(scores)),
    }


def render_dashboard() -> None:
    """Render the RecruitAI Streamlit dashboard."""
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install requirements.txt to launch the dashboard.")

    settings = get_settings()
    st.set_page_config(page_title=settings.project_name, layout="wide")
    st.title("RecruitAI")
    st.caption("Offline AI candidate ranking")
    top_k = st.slider("Candidates to display", min_value=10, max_value=100, value=25, step=5)
    ranked = load_dashboard_results(top_k=top_k)
    metrics = dashboard_metrics(ranked)
    column_a, column_b, column_c = st.columns(3)
    column_a.metric("Candidates", int(metrics["candidate_count"]))
    column_b.metric("Average Score", f"{metrics['average_score']:.3f}")
    column_c.metric("Top Score", f"{metrics['top_score']:.3f}")
    st.dataframe(ranked, use_container_width=True)
    if px is not None and ranked:
        figure = px.bar(ranked[: min(20, len(ranked))], x="candidate_id", y="score", title="Top Candidate Scores")
        st.plotly_chart(figure, use_container_width=True)

