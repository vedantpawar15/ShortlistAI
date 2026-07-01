"""Streamlit dashboard for RecruitAI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recruit_ai.data.sample_data import sample_candidates, sample_job
from recruit_ai.pipeline import RecruitAIRanker

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - exercised only when streamlit is absent
    st = None


def build_dashboard() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install dependencies from requirements.txt before running `streamlit run app.py`."
        )

    ranker = RecruitAIRanker()
    default_job = sample_job()
    default_candidates = sample_candidates()

    st.set_page_config(page_title="RecruitAI", page_icon=":mag:", layout="wide")
    st.title("RecruitAI")
    st.caption("Interview-grade recruiter ranking demo with semantic retrieval, hybrid scoring, and explainability.")

    col_left, col_right = st.columns([1, 1.35])
    with col_left:
        st.subheader("Job Definition")
        job_title = st.text_input("Title", value=default_job.title)
        job_summary = st.text_area("Summary", value=default_job.summary, height=180)
        required_skills = st.text_area(
            "Required skills",
            value=", ".join(default_job.required_skills),
            help="Comma-separated required skills.",
        )
        preferred_skills = st.text_area(
            "Preferred skills",
            value=", ".join(default_job.preferred_skills),
            help="Comma-separated preferred skills.",
        )
        min_years = st.slider("Minimum years of experience", min_value=0, max_value=15, value=int(default_job.minimum_years_experience))
        location = st.text_input("Location", value=default_job.location)

    with col_right:
        st.subheader("Candidate Dataset")
        dataset_json = st.text_area(
            "Candidates JSON",
            value=json.dumps([candidate.to_dict() for candidate in default_candidates], indent=2),
            height=420,
        )

    job_payload = {
        "job_id": default_job.job_id,
        "title": job_title,
        "summary": job_summary,
        "required_skills": [item.strip() for item in required_skills.split(",") if item.strip()],
        "preferred_skills": [item.strip() for item in preferred_skills.split(",") if item.strip()],
        "minimum_years_experience": float(min_years),
        "location": location,
        "responsibilities": default_job.responsibilities,
        "keywords": default_job.keywords,
    }

    try:
        candidate_payloads = json.loads(dataset_json)
        results = ranker.rank_from_dicts(job_payload, candidate_payloads)
    except Exception as exc:  # pragma: no cover - UI-only safety path
        st.error(f"Unable to rank candidates: {exc}")
        return

    st.subheader("Ranked Shortlist")
    for index, result in enumerate(results, start=1):
        candidate = result["candidate"]
        with st.container(border=True):
            score_col, detail_col = st.columns([0.25, 0.75])
            with score_col:
                st.metric(label=f"Rank #{index}", value=f"{result['total_score']:.3f}")
                st.write(candidate["name"])
                st.caption(candidate["headline"])
            with detail_col:
                st.write("Matched required skills:", ", ".join(result["matched_required_skills"]) or "None")
                st.write("Missing required skills:", ", ".join(result["missing_required_skills"]) or "None")
                st.write("Explanation:")
                for line in result["explanation"]:
                    st.write(f"- {line}")
                st.json(result["feature_scores"])


if __name__ == "__main__":
    build_dashboard()
