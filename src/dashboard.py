"""Streamlit dashboard for ShortlistAI recruiter workflows.

Layout
------
Sidebar  — Configuration sliders and controls.
Tab 1    — Overview: KPI metrics, top candidates bar chart, score histogram.
Tab 2    — Candidates: Filterable ranked table with inline score breakdowns.
Tab 3    — Analysis: Skill gap heatmap, score component radar chart.
Tab 4    — About: Pipeline description and legend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    import streamlit as st

    _STREAMLIT = True
except ImportError:
    _STREAMLIT = False  # type: ignore[assignment]

from src.config import get_settings
from src.data_loader import DataLoader
from src.ranking_engine import RankingEngine


# ---------------------------------------------------------------------------
# Data helpers (pure functions — testable without Streamlit)
# ---------------------------------------------------------------------------


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


def score_tier(score: float) -> str:
    """Assign a human-readable tier label to a composite score."""
    if score >= 0.75:
        return "🟢 Strong"
    if score >= 0.50:
        return "🟡 Moderate"
    return "🔴 Weak"


# ---------------------------------------------------------------------------
# Chart builders (return Plotly figures — testable without Streamlit)
# ---------------------------------------------------------------------------

_SCORE_COLUMNS = [
    "semantic_similarity",
    "skill_overlap",
    "experience_match",
    "education_match",
    "title_similarity",
    "location_match",
    "behavioral_signal_score",
]

_SCORE_LABELS = {
    "semantic_similarity": "Semantic",
    "skill_overlap": "Skills",
    "experience_match": "Experience",
    "education_match": "Education",
    "title_similarity": "Title",
    "location_match": "Location",
    "behavioral_signal_score": "Behavior",
}


def build_top_candidates_chart(ranked: list[dict[str, Any]], top_n: int = 20) -> Any:
    """Horizontal bar chart of top-N candidate scores with tier colouring."""
    if not _PLOTLY:
        return None
    df = pd.DataFrame(ranked[:top_n])
    df["score"] = df["score"].astype(float)
    df["tier"] = df["score"].apply(score_tier)
    color_map = {"🟢 Strong": "#10b981", "🟡 Moderate": "#f59e0b", "🔴 Weak": "#ef4444"}
    fig = px.bar(
        df,
        x="score",
        y="candidate_id",
        orientation="h",
        color="tier",
        color_discrete_map=color_map,
        title=f"Top {min(top_n, len(ranked))} Candidates by Composite Score",
        labels={"score": "Composite Score", "candidate_id": "Candidate ID"},
        text="score",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        legend_title="Tier",
        height=max(300, top_n * 28),
        margin={"l": 10, "r": 60, "t": 50, "b": 40},
    )
    return fig


def build_score_histogram(ranked: list[dict[str, Any]]) -> Any:
    """Distribution of composite scores across all candidates."""
    if not _PLOTLY:
        return None
    scores = [float(r.get("score", 0.0) or 0.0) for r in ranked]
    fig = px.histogram(
        x=scores,
        nbins=20,
        title="Score Distribution Across Applicant Pool",
        labels={"x": "Composite Score", "count": "Count"},
        color_discrete_sequence=["#6366f1"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        bargap=0.05,
    )
    return fig


def build_skill_gap_heatmap(ranked: list[dict[str, Any]], top_n: int = 15) -> Any:
    """Heatmap showing which candidates cover which dimension."""
    if not _PLOTLY:
        return None
    available_cols = [c for c in _SCORE_COLUMNS if any(c in str(r) for r in ranked[:1])]
    if not available_cols:
        return None
    df = pd.DataFrame(ranked[:top_n])
    present_cols = [c for c in _SCORE_COLUMNS if c in df.columns]
    if not present_cols:
        return None
    heat_df = df[["candidate_id"] + present_cols].set_index("candidate_id")
    heat_df = heat_df.astype(float).fillna(0.0)
    heat_df.columns = [_SCORE_LABELS.get(c, c) for c in heat_df.columns]
    fig = px.imshow(
        heat_df,
        color_continuous_scale="RdYlGn",
        zmin=0.0,
        zmax=1.0,
        title=f"Score Component Heatmap (Top {min(top_n, len(ranked))} Candidates)",
        aspect="auto",
        text_auto=".2f",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        coloraxis_colorbar_title="Score",
    )
    return fig


def build_radar_chart(row: dict[str, Any]) -> Any:
    """Radar / spider chart showing the score component breakdown for one candidate."""
    if not _PLOTLY:
        return None
    present_dims = [(lbl, row.get(col, 0.0) or 0.0) for col, lbl in _SCORE_LABELS.items() if col in row]
    if not present_dims:
        return None
    labels, values = zip(*present_dims)
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=list(values) + [values[0]],
            theta=list(labels) + [labels[0]],
            fill="toself",
            fillcolor="rgba(99,102,241,0.3)",
            line_color="#6366f1",
            name="Score Components",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#475569"),
            angularaxis=dict(gridcolor="#475569"),
            bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        title=f"Score Breakdown — {row.get('candidate_id', '')}",
        showlegend=False,
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
<style>
    /* Dark professional theme */
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .stApp header { background-color: #0f172a; }
    .block-container { padding: 1.5rem 2rem; }
    h1 { color: #818cf8; font-family: 'Inter', sans-serif; font-weight: 800; }
    h2, h3 { color: #c7d2fe; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .stMetric label { color: #94a3b8 !important; font-size: 0.8rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #818cf8 !important; font-size: 1.8rem !important; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #818cf8; border-bottom-color: #818cf8 !important; }
    .stSlider [data-baseweb="slider"] > div { background: #6366f1; }
    .stSidebar { background-color: #1e293b; }
    .tier-strong { color: #10b981; font-weight: 600; }
    .tier-moderate { color: #f59e0b; font-weight: 600; }
    .tier-weak { color: #ef4444; font-weight: 600; }
</style>
"""


def _apply_css() -> None:
    if _STREAMLIT:
        st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def render_dashboard() -> None:
    """Render the ShortlistAI Streamlit dashboard."""
    if not _STREAMLIT:
        raise RuntimeError(
            "Streamlit is not installed. Install requirements.txt to launch the dashboard."
        )

    settings = get_settings()
    st.set_page_config(
        page_title=f"{settings.project_name} — AI Candidate Ranking",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_css()

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        top_k = st.slider("Candidates to display", min_value=5, max_value=100, value=25, step=5)
        top_bar_n = st.slider("Top-N bar chart", min_value=5, max_value=30, value=15, step=5)
        score_threshold = st.slider("Min score filter", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        st.markdown("---")
        st.markdown(
            "**ShortlistAI** — offline-first candidate ranking using BGE embeddings, "
            "FAISS retrieval, and a hybrid feature-scoring pipeline."
        )

    # ---- Header ----
    st.markdown("# 🎯 ShortlistAI")
    st.markdown(
        "<p style='color:#94a3b8;margin-top:-0.8rem;'>AI-powered candidate ranking · "
        f"Project: <b style='color:#818cf8'>{settings.project_name}</b></p>",
        unsafe_allow_html=True,
    )

    # ---- Load data ----
    with st.spinner("Loading ranked candidates…"):
        try:
            ranked = load_dashboard_results(top_k=top_k)
        except Exception as exc:
            st.error(f"❌ Failed to load results: {exc}")
            return

    # Apply score filter.
    ranked = [r for r in ranked if float(r.get("score", 0.0) or 0.0) >= score_threshold]

    if not ranked:
        st.warning("No candidates match the current filter criteria.")
        return

    metrics = dashboard_metrics(ranked)

    # ---- KPI Row ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Candidates", int(metrics["candidate_count"]))
    col2.metric("📊 Avg Score", f"{metrics['average_score']:.3f}")
    col3.metric("🏆 Top Score", f"{metrics['top_score']:.3f}")
    strong_count = sum(1 for r in ranked if float(r.get("score", 0.0) or 0.0) >= 0.75)
    col4.metric("🟢 Strong Fit", f"{strong_count} ({100*strong_count//max(len(ranked),1)}%)")

    st.markdown("---")

    # ---- Tabs ----
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "👤 Candidates", "🔬 Analysis", "📖 About"])

    # TAB 1 — Overview
    with tab1:
        chart_col, hist_col = st.columns([3, 2])
        with chart_col:
            fig = build_top_candidates_chart(ranked, top_n=top_bar_n)
            if fig and _PLOTLY:
                st.plotly_chart(fig, use_container_width=True)
        with hist_col:
            hist_fig = build_score_histogram(ranked)
            if hist_fig and _PLOTLY:
                st.plotly_chart(hist_fig, use_container_width=True)

    # TAB 2 — Candidates
    with tab2:
        display_cols = ["candidate_id", "rank", "score", "reasoning"]
        df = pd.DataFrame(ranked)
        available_display = [c for c in display_cols if c in df.columns]
        df["tier"] = df["score"].astype(float).apply(score_tier)
        show_cols = ["tier"] + available_display
        st.dataframe(
            df[[c for c in show_cols if c in df.columns]].style.background_gradient(
                subset=["score"] if "score" in df.columns else [], cmap="RdYlGn", vmin=0, vmax=1
            ),
            use_container_width=True,
            height=420,
        )

        # Export button.
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export to CSV",
            data=csv_data,
            file_name="ranked_candidates.csv",
            mime="text/csv",
        )

        # Candidate detail panel.
        candidate_ids = [str(r.get("candidate_id", "")) for r in ranked]
        selected_id = st.selectbox("🔍 Candidate detail", options=candidate_ids)
        selected_row = next((r for r in ranked if str(r.get("candidate_id")) == selected_id), None)
        if selected_row:
            with st.expander(f"📋 {selected_id} — Full Detail", expanded=True):
                detail_col, radar_col = st.columns([1, 1])
                with detail_col:
                    st.markdown(f"**Rank:** {selected_row.get('rank', '—')}")
                    st.markdown(f"**Score:** `{float(selected_row.get('score', 0)):.4f}` {score_tier(float(selected_row.get('score', 0)))}")
                    st.markdown(f"**Reasoning:** {selected_row.get('reasoning', '—')}")
                    score_cols = [c for c in _SCORE_COLUMNS if c in selected_row]
                    if score_cols:
                        st.markdown("**Score Components:**")
                        for col in score_cols:
                            val = float(selected_row.get(col) or 0.0)
                            label = _SCORE_LABELS.get(col, col)
                            st.progress(val, text=f"{label}: {val:.3f}")
                with radar_col:
                    radar_fig = build_radar_chart(selected_row)
                    if radar_fig and _PLOTLY:
                        st.plotly_chart(radar_fig, use_container_width=True)

    # TAB 3 — Analysis
    with tab3:
        heatmap_n = min(20, len(ranked))
        heatmap_fig = build_skill_gap_heatmap(ranked, top_n=heatmap_n)
        if heatmap_fig and _PLOTLY:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info("Score component columns not found in results. Run `python main.py rank` to generate detailed scores.")

    # TAB 4 — About
    with tab4:
        st.markdown("""
## Pipeline Architecture

```
DataLoader → TextPreprocessor → EmbeddingModel (BGE-small)
    → FaissVectorStore (semantic retrieval)
    → FeatureEngineer (lexical + BM25 + skill + experience + education + location + behavior)
    → CandidateScorer (weighted sum + hybrid blending)
    → Reranker (cross-encoder or RRF)
    → RankingExplainer (batch-calibrated explanations)
    → submission.csv
```

## Score Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Semantic Similarity | 30% | BGE embedding cosine similarity |
| Skill Overlap | 25% | Required skill coverage |
| Experience Match | 15% | Years vs requirement |
| Education Match | 5% | Degree relevance |
| Title Similarity | 10% | Current title vs job title |
| Location Match | 5% | Geographic affinity |
| Behavioral Signals | 10% | Open-to-work, GitHub activity, etc. |

## Running Commands

```bash
python main.py health         # Check configuration
python main.py rank           # Generate submission.csv
streamlit run app.py          # Launch this dashboard
```
        """)
