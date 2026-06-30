"""Feature engineering for candidate-job matching."""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    class MinMaxScaler:
        """Fallback MinMaxScaler for environments without scikit-learn."""

        def fit_transform(self, values: np.ndarray) -> np.ndarray:
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            if maximum == minimum:
                return np.zeros_like(values, dtype=float)
            return (values - minimum) / (maximum - minimum)

from src.preprocessing import DEFAULT_SKILL_ALIASES, SkillNormalizer, split_skill_text


class FeatureEngineer:
    """Create model-ready ranking features from extracted signals."""

    def __init__(self, skill_normalizer: SkillNormalizer | None = None) -> None:
        self.scaler = MinMaxScaler()
        self.skill_normalizer = skill_normalizer or SkillNormalizer()

    def build_features(self, candidates: pd.DataFrame, job: pd.Series) -> pd.DataFrame:
        """Build ranking features for candidates against one job."""
        job_record = job.to_dict() if hasattr(job, "to_dict") else dict(job)
        job_text = document_text(job_record)
        job_tokens = tokenize(job_text)
        job_skills = extract_skills(job_text, self.skill_normalizer)
        required_experience = extract_experience_years(job)
        job_location = first_present(job_record, ("location",))
        job_title = first_present(job_record, ("title", "job_title"))
        rows: list[dict[str, Any]] = []

        for index, candidate in candidates.reset_index(drop=True).iterrows():
            candidate_record = candidate.to_dict()
            candidate_text = document_text(candidate_record)
            candidate_tokens = tokenize(candidate_text)
            candidate_skills = extract_candidate_skills(candidate_record, candidate_text, self.skill_normalizer)
            candidate_experience = extract_experience_years(candidate)
            candidate_title = first_present(
                candidate_record,
                (
                    "profile_current_title",
                    "current_title",
                    "title",
                    "profile_headline",
                    "headline",
                ),
            )

            semantic_similarity = lexical_similarity(candidate_tokens, job_tokens)
            skill_overlap = overlap_score(candidate_skills, job_skills)
            experience_match = experience_score(candidate_experience, required_experience)
            education_match = education_score(candidate_record, job_text)
            title_similarity = title_match_score(candidate_title, job_title)
            location_match = location_score(candidate_record, job_location)
            behavioral_signal_score = behavioral_score(candidate_record)

            rows.append(
                {
                    "candidate_id": candidate_record.get("candidate_id", f"candidate_{index}"),
                    "candidate_index": index,
                    "semantic_similarity": semantic_similarity,
                    "skill_overlap": skill_overlap,
                    "experience_match": experience_match,
                    "education_match": education_match,
                    "title_similarity": title_similarity,
                    "location_match": location_match,
                    "behavioral_signal_score": behavioral_signal_score,
                    "candidate_experience_years": candidate_experience,
                    "required_experience_years": required_experience,
                    "matched_skill_count": len(candidate_skills & job_skills),
                    "missing_skill_count": len(job_skills - candidate_skills),
                    "matched_skills": sorted(candidate_skills & job_skills),
                    "missing_skills": sorted(job_skills - candidate_skills),
                    "candidate_title": candidate_title,
                }
            )

        return pd.DataFrame(rows)

    def normalize_scores(self, values: np.ndarray) -> np.ndarray:
        """Normalize numeric score columns into a comparable range."""
        if values.size == 0:
            return values
        return self.scaler.fit_transform(values.reshape(-1, 1)).ravel()


def document_text(record: dict[str, Any]) -> str:
    """Build a searchable text representation from a normalized record."""
    preferred = (
        "semantic_document",
        "document",
        "cleaned_document",
        "description",
        "job_description",
        "profile_summary",
        "summary",
        "profile_headline",
        "headline",
    )
    parts = [str(record[key]) for key in preferred if key in record and is_present(record[key])]
    if parts:
        return " ".join(parts)
    return " ".join(str(value) for value in record.values() if is_present(value))


def tokenize(text: str) -> set[str]:
    """Tokenize text into informative lowercase words."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", text.lower())
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "for",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    return {word for word in words if len(word) > 1 and word not in stop_words}


def lexical_similarity(candidate_tokens: set[str], job_tokens: set[str]) -> float:
    """Combine token overlap with a small fuzzy fallback for sparse text."""
    if not candidate_tokens or not job_tokens:
        return 0.0
    overlap = len(candidate_tokens & job_tokens) / len(job_tokens)
    candidate_text = " ".join(sorted(candidate_tokens))
    job_text = " ".join(sorted(job_tokens))
    fuzzy = SequenceMatcher(None, candidate_text, job_text).ratio()
    return float(min(1.0, (0.8 * overlap) + (0.2 * fuzzy)))


def extract_candidate_skills(record: dict[str, Any], text: str, normalizer: SkillNormalizer) -> set[str]:
    """Extract candidate skills from explicit fields and profile text."""
    skills: list[object] = []
    raw_skills = record.get("skills")
    if isinstance(raw_skills, str):
        skills.extend(split_skill_text(raw_skills))
    elif isinstance(raw_skills, Iterable):
        skills.extend(raw_skills)
    skills.extend(value for key, value in record.items() if re.fullmatch(r"skills_\d+_name", str(key)) and value)
    return set(normalizer.normalize_many(skills)) | extract_skills(text, normalizer)


def extract_skills(text: str, normalizer: SkillNormalizer) -> set[str]:
    """Extract known skill aliases from text."""
    normalized_text = " ".join(tokenize(text))
    found = {
        canonical
        for alias, canonical in DEFAULT_SKILL_ALIASES.items()
        if re.search(rf"\b{re.escape(alias)}\b", normalized_text)
    }
    return set(normalizer.normalize_many(found))


def overlap_score(candidate_skills: set[str], job_skills: set[str]) -> float:
    """Compute required-skill coverage."""
    if not job_skills:
        return 0.5
    return len(candidate_skills & job_skills) / len(job_skills)


def extract_experience_years(record: pd.Series | dict[str, Any]) -> float | None:
    """Return the best available years-of-experience signal."""
    keys = (
        "profile_years_of_experience",
        "years_of_experience",
        "experience_years",
        "total_experience_years",
    )
    getter = record.get
    for key in keys:
        value = getter(key)
        if not is_present(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    text = document_text(record.to_dict() if hasattr(record, "to_dict") else dict(record))
    match = re.search(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", text.lower())
    return float(match.group(1)) if match else None


def experience_score(candidate_years: float | None, required_years: float | None) -> float:
    """Score experience as coverage of the job requirement."""
    if required_years is None or required_years <= 0:
        return 0.5 if candidate_years is None else min(1.0, candidate_years / 10.0)
    if candidate_years is None:
        return 0.0
    return float(min(1.0, candidate_years / required_years))


def education_score(record: dict[str, Any], job_text: str) -> float:
    """Reward candidates with relevant education when the job mentions education."""
    education_text = " ".join(
        str(value)
        for key, value in record.items()
        if str(key).startswith("education") and is_present(value)
    ).lower()
    if not education_text:
        return 0.0
    job_lower = job_text.lower()
    if not any(term in job_lower for term in ("degree", "bachelor", "master", "phd", "computer science")):
        return 0.5
    return 1.0 if any(term in education_text for term in ("computer", "data", "ai", "ml", "science")) else 0.5


def title_match_score(candidate_title: str, job_title: str) -> float:
    """Score how similar the current candidate title is to the job title."""
    if not candidate_title or not job_title:
        return 0.0
    return float(SequenceMatcher(None, candidate_title.lower(), job_title.lower()).ratio())


def location_score(record: dict[str, Any], job_location: str) -> float:
    """Score candidate location affinity to the job location."""
    if not job_location:
        return 0.5
    candidate_location = ""
    for key in ("location", "profile_location", "current_location"):
        value = record.get(key)
        if is_present(value) and str(value).strip():
            candidate_location = str(value)
            break
    if not candidate_location:
        return 0.0
    return 1.0 if candidate_location.lower() in job_location.lower() or job_location.lower() in candidate_location.lower() else 0.2


def behavioral_score(record: dict[str, Any]) -> float:
    """Aggregate recruiter-signal fields into a bounded numeric score."""
    numeric_values: list[float] = []
    boolean_bonus = 0.0
    for key, value in record.items():
        key_text = str(key)
        if not key_text.startswith("redrob_signals_"):
            continue
        if isinstance(value, bool):
            boolean_bonus += 1.0 if value else 0.0
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric_values and boolean_bonus == 0.0:
        return 0.0
    normalized_numeric = sum(min(max(value, 0.0), 100.0) / 100.0 for value in numeric_values) / max(len(numeric_values), 1)
    return float(min(1.0, normalized_numeric + (0.1 * boolean_bonus)))


def first_present(record: dict[str, Any], keys: Iterable[str]) -> str:
    """Return the first meaningful text value for a set of keys."""
    for key in keys:
        value = record.get(key)
        if is_present(value) and str(value).strip():
            return str(value)
    return "Candidate"


def is_present(value: Any) -> bool:
    """Return True for scalar or container values that carry useful content."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    try:
        return bool(pd.notna(value))
    except (TypeError, ValueError):
        return True

