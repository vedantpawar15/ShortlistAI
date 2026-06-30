"""Streamlit dashboard shell for recruiter workflows."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.config import get_settings


def render_dashboard() -> None:
    """Render the RecruitAI Streamlit dashboard."""
    settings = get_settings()
    st.set_page_config(page_title=settings.project_name, layout="wide")
    st.title("RecruitAI")
    st.caption("Offline candidate ranking architecture scaffold.")
    st.info("Ranking functionality is not implemented yet.")

    placeholder_data = {"stage": ["Architecture"], "status": [1]}
    figure = px.bar(placeholder_data, x="stage", y="status", title="Project Readiness")
    st.plotly_chart(figure, use_container_width=True)

