"""Tests for semantic candidate preprocessing."""

from __future__ import annotations

import pandas as pd

from src.preprocessing import SkillNormalizer, TextPreprocessor, split_skill_text


def test_text_normalization_lowercase_unicode_punctuation_and_whitespace() -> None:
    preprocessor = TextPreprocessor()

    cleaned = preprocessor.preprocess("  Café—ML, NLP!!!\n\tEngineer  ")

    assert cleaned == "cafe ml nlp engineer"


def test_skill_normalization_aliases_and_fuzzy_mapping() -> None:
    normalizer = SkillNormalizer(fuzzy_threshold=80)

    skills = normalizer.normalize_many(["AI", "JS", "ml", "Py", "K8S", "Tensorflw"])

    assert "artificial intelligence" in skills
    assert "javascript" in skills
    assert "machine learning" in skills
    assert "python" in skills
    assert "kubernetes" in skills
    assert "tensorflow" in skills


def test_build_candidate_profile_from_nested_candidate() -> None:
    candidate = {
        "candidate_id": "CAND_1",
        "profile": {
            "headline": "Backend Engineer | JS, ML",
            "summary": "Builds APIs and ML features.",
            "current_title": "Senior Engineer",
            "current_company": "Acme AI",
            "current_industry": "SaaS",
            "years_of_experience": 7.5,
        },
        "career_history": [
            {
                "company": "Acme",
                "title": "Engineer",
                "industry": "Cloud",
                "description": "Built k8s services and TF inference jobs.",
            }
        ],
        "skills": [
            {"name": "JS", "proficiency": "advanced"},
            {"name": "ml", "proficiency": "intermediate"},
            {"name": "Py", "proficiency": "advanced"},
        ],
        "education": [{"institution": "State University", "degree": "B.Tech", "field_of_study": "Computer Science"}],
        "certifications": [{"name": "Cloud Architect", "issuer": "Vendor", "year": 2024}],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "open_to_work_flag": True,
            "github_activity_score": 82,
            "preferred_work_mode": "remote",
        },
    }

    profile = TextPreprocessor().build_candidate_profile(candidate)

    assert profile.candidate_id == "CAND_1"
    assert profile.normalized_skills == ["javascript", "machine learning", "python"]
    assert "summary builds apis and ml features" in profile.document
    assert "career history engineer acme cloud built k8s services and tf inference jobs" in profile.document
    assert "skills javascript machine learning python" in profile.document
    assert "education state university b tech computer science" in profile.document
    assert "certifications cloud architect vendor 2024" in profile.document
    assert "languages english professional" in profile.document
    assert (
        "behavior signals github activity score 82 open to work flag true preferred work mode remote"
        in profile.document
    )
    assert "experience 7 5" in profile.document
    assert "current role senior engineer" in profile.document
    assert "current company acme ai" in profile.document


def test_build_candidate_profile_from_flattened_dataframe_row() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "CAND_2",
                "profile_summary": "NLP engineer working on search.",
                "profile_headline": "ML Engineer",
                "profile_years_of_experience": 4,
                "profile_current_title": "Applied AI Engineer",
                "profile_current_company": "SearchCo",
                "career_history_0_title": "Applied Scientist",
                "career_history_0_company": "SearchCo",
                "career_history_0_description": "Built ranking models.",
                "skills_0_name": "nlp",
                "skills_1_name": "k8s",
                "education_0_degree": "M.S.",
                "education_0_field_of_study": "AI",
                "certifications_0_name": "TensorFlow Developer",
                "languages_0_language": "Hindi",
                "languages_0_proficiency": "native",
                "redrob_signals_open_to_work_flag": True,
            }
        ]
    )

    profiles = TextPreprocessor().build_candidate_profiles(frame)

    assert len(profiles) == 1
    assert profiles[0].normalized_skills == ["natural language processing", "kubernetes"]
    assert "headline ml engineer" in profiles[0].document
    assert "career history applied scientist searchco built ranking models" in profiles[0].document
    assert "skills natural language processing kubernetes" in profiles[0].document
    assert "behavior signals open to work flag true" in profiles[0].document
    assert "experience 4" in profiles[0].document
    assert "current role applied ai engineer" in profiles[0].document
    assert "current company searchco" in profiles[0].document


def test_preprocess_candidates_returns_cleaned_documents() -> None:
    candidates = [{"candidate_id": "CAND_3", "profile_summary": "Expert in JS, NLP, and TF."}]

    documents = TextPreprocessor().preprocess_candidates(candidates)

    assert documents == ["summary expert in js nlp and tf"]


def test_split_skill_text_handles_delimiters_and_json_lists() -> None:
    assert split_skill_text("Python | ML, SQL; NLP") == ["Python", "ML", "SQL", "NLP"]
    assert split_skill_text('[{"name": "JS"}, {"name": "TF"}]') == ["JS", "TF"]
