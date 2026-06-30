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


def render_dashboard() -> None:
    """Render the RecruitAI Streamlit dashboard."""
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install requirements.txt to launch the dashboard.")

    settings = get_settings()
    st.set_page_config(page_title=settings.project_name, layout="wide")
    st.title("RecruitAI")
    st.caption("Offline AI candidate ranking")

    submission_path = Path("submission.csv")
    if submission_path.exists():
        ranked = pd.read_csv(submission_path).to_dict(orient="records")
    else:
        loader = DataLoader(settings.data_dir)
        candidates = loader.load_candidates("sample_candidates.json")
        jobs = loader.load_jobs("job_description.docx")
        ranked_frame = RankingEngine().rank_candidates(candidates, jobs.iloc[0]).head(100)
        ranked = ranked_frame.to_dict(orient="records")

    st.dataframe(ranked, use_container_width=True)
    if px is not None and ranked:
        figure = px.bar(ranked[:20], x="candidate_id", y="score", title="Top Candidate Scores")
        st.plotly_chart(figure, use_container_width=True)

