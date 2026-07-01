"""Feature engineering for candidate-job matching.

Design notes
------------
* ``semantic_similarity``  — Cosine similarity from the FAISS/embedding retrieval stage.
  Stored as ``_semantic_score`` in the retrieved candidates frame; falls back to
  ``lexical_similarity`` when embeddings were not used.
* ``lexical_similarity``   — Deterministic token-overlap + fuzzy-string signal derived
  entirely from text without a neural model.
* All other columns are self-describing scalar features in [0, 1].

Performance notes
-----------------
* Job-side tokens and skills are computed ONCE outside the candidate loop.
* Candidate skill extraction reuses flattened keys without redundant tokenisation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:

    class MinMaxScaler:  # type: ignore[no-redef]
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
        """Build ranking features for candidates against one job.

        If the candidates frame contains a ``_semantic_score`` column produced
        by the FAISS retrieval stage, that cosine similarity is used as
        ``semantic_similarity``.  Otherwise ``lexical_similarity`` is promoted.
        """
        job_record = job.to_dict() if hasattr(job, "to_dict") else dict(job)
        job_text = document_text(job_record)
        job_tokens = tokenize(job_text)
        job_skills = extract_skills(job_text, self.skill_normalizer)
        required_experience = extract_experience_years(job)
        job_location = first_present(job_record, ("location",))
        job_title = first_present(job_record, ("title", "job_title"))

        # Pre-compute job TF-IDF vector once for all candidates.
        job_tf = term_frequencies(job_tokens)

        has_semantic_scores = "_semantic_score" in candidates.columns
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

            lex_sim = lexical_similarity(candidate_tokens, job_tokens)
            bm25_sim = bm25_similarity(candidate_tokens, job_tf, len(job_tokens))

            # ``semantic_similarity`` is the embedding cosine score when available;
            # otherwise we promote the best available lexical signal.
            if has_semantic_scores:
                semantic_score = float(candidate_record.get("_semantic_score") or 0.0)
            else:
                semantic_score = max(lex_sim, bm25_sim)

            is_honeypot = detect_honeypot(candidate_record, candidate_experience)
            rich_skills = extract_rich_skills(candidate_record, self.skill_normalizer)
            skill_overlap = overlap_score(candidate_skills, job_skills, rich_skills)
            experience_match = experience_score(candidate_experience, required_experience)
            education_match = education_score(candidate_record, job_text)
            title_similarity = title_match_score(candidate_title, job_title)
            location_match = location_score(candidate_record, job_location)
            behavioral_signal_score = behavioral_score(candidate_record)

            rows.append(
                {
                    "candidate_id": candidate_record.get("candidate_id", f"candidate_{index}"),
                    "candidate_index": index,
                    "semantic_similarity": semantic_score,
                    "lexical_similarity": lex_sim,
                    "bm25_similarity": bm25_sim,
                    "skill_overlap": skill_overlap,
                    "experience_match": experience_match,
                    "education_match": education_match,
                    "title_similarity": title_similarity,
                    "location_match": location_match,
                    "behavioral_signal_score": behavioral_signal_score,
                    "is_honeypot": is_honeypot,
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


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


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


def term_frequencies(tokens: set[str]) -> dict[str, float]:
    """Compute normalized term frequencies from a token set."""
    if not tokens:
        return {}
    tf = 1.0 / len(tokens)
    return {token: tf for token in tokens}


def bm25_similarity(
    candidate_tokens: set[str],
    job_tf: dict[str, float],
    job_length: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """BM25-inspired similarity between candidate tokens and pre-computed job TF.

    Uses a simplified BM25 formulation: instead of a full corpus IDF we treat
    each job token as equally important (IDF=1) and apply the BM25 length
    normalisation factor.
    """
    if not candidate_tokens or not job_tf:
        return 0.0
    avg_doc_len = max(job_length, 1)
    score = 0.0
    for token, tf_q in job_tf.items():
        tf_d = 1.0 if token in candidate_tokens else 0.0
        numerator = tf_d * (k1 + 1)
        denominator = tf_d + k1 * (1 - b + b * len(candidate_tokens) / avg_doc_len)
        score += tf_q * (numerator / max(denominator, 1e-9))
    # Normalise to [0, 1] using the theoretical max (all job tokens present).
    max_score = sum(
        tf_q * (k1 + 1) / (1 + k1 * (1 - b + b))
        for tf_q in job_tf.values()
    )
    return float(min(1.0, score / max(max_score, 1e-9)))


def lexical_similarity(candidate_tokens: set[str], job_tokens: set[str]) -> float:
    """Combine token overlap with a small fuzzy fallback for sparse text."""
    if not candidate_tokens or not job_tokens:
        return 0.0
    overlap = len(candidate_tokens & job_tokens) / len(job_tokens)
    candidate_text = " ".join(sorted(candidate_tokens))
    job_text = " ".join(sorted(job_tokens))
    fuzzy = SequenceMatcher(None, candidate_text, job_text).ratio()
    return float(min(1.0, (0.8 * overlap) + (0.2 * fuzzy)))


# ---------------------------------------------------------------------------
# Skill helpers
# ---------------------------------------------------------------------------


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


def extract_rich_skills(record: dict[str, Any], normalizer: SkillNormalizer) -> dict[str, float]:
    """Extract skills and assign them a weight based on proficiency, duration and endorsements."""
    prof_weights = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}
    rich_skills: dict[str, float] = {}
    
    for i in range(100):
        name_key = f"skills_{i}_name"
        if name_key not in record:
            if i > 5 and not any(k.startswith(f"skills_{i}") for k in record):
                break
            continue
            
        name_val = record[name_key]
        if not name_val:
            continue
            
        canonical_names = normalizer.normalize_many([str(name_val)])
        if not canonical_names:
            continue
            
        prof = str(record.get(f"skills_{i}_proficiency", "")).lower()
        prof_weight = prof_weights.get(prof, 0.5)
        
        try:
            dur = float(record.get(f"skills_{i}_duration_months") or 0.0)
        except (TypeError, ValueError):
            dur = 0.0
            
        try:
            endorse = float(record.get(f"skills_{i}_endorsements") or 0.0)
        except (TypeError, ValueError):
            endorse = 0.0
            
        dur_weight = min(1.0, dur / 24.0) if dur > 0 else 0.1
        endorse_weight = min(1.0, endorse / 10.0)
        
        skill_quality = (prof_weight * 0.4) + (dur_weight * 0.4) + (endorse_weight * 0.2)
        
        for c in canonical_names:
            rich_skills[c] = max(rich_skills.get(c, 0.0), skill_quality)
            
    return rich_skills


def overlap_score(candidate_skills: set[str], job_skills: set[str], rich_skills: dict[str, float] | None = None) -> float:
    """Compute required-skill coverage weighted by skill quality."""
    if not job_skills:
        return 0.5
    if rich_skills is None:
        rich_skills = {}
        
    overlap_val = 0.0
    for job_skill in job_skills:
        if job_skill in candidate_skills:
            quality = rich_skills.get(job_skill, 0.5)
            overlap_val += quality
            
    return overlap_val / len(job_skills)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


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
    """Reward candidates with relevant education when the job mentions education.

    Improvements over original:
    * Tiered scoring (1.0 / 0.75 / 0.5) based on degree relevance depth.
    * Recognises more relevant fields of study.
    """
    education_text = " ".join(
        str(value)
        for key, value in record.items()
        if str(key).startswith("education") and is_present(value)
    ).lower()

    if not education_text:
        return 0.0

    job_lower = job_text.lower()
    job_requires_education = any(
        term in job_lower
        for term in ("degree", "bachelor", "master", "phd", "computer science", "engineering")
    )
    if not job_requires_education:
        return 0.5

    highly_relevant = ("computer", "data", "ai", "ml", "machine learning", "software", "information")
    partially_relevant = ("science", "mathematics", "statistics", "physics", "engineering")

    if any(term in education_text for term in highly_relevant):
        return 1.0
    if any(term in education_text for term in partially_relevant):
        return 0.75
    return 0.5


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
    return (
        1.0
        if candidate_location.lower() in job_location.lower()
        or job_location.lower() in candidate_location.lower()
        else 0.2
    )


def detect_honeypot(record: dict[str, Any], candidate_experience: float | None) -> bool:
    """Flag candidates with impossible profiles."""
    for i in range(100):
        prof_key = f"skills_{i}_proficiency"
        dur_key = f"skills_{i}_duration_months"
        if prof_key in record:
            prof = str(record[prof_key]).lower()
            if prof in ("advanced", "expert"):
                try:
                    dur = float(record.get(dur_key) or 0.0)
                    if dur == 0.0:
                        return True
                except (TypeError, ValueError):
                    pass
        else:
            if i > 5 and not any(k.startswith(f"skills_{i}") for k in record):
                break
                
    total_career_months = 0.0
    for i in range(100):
        dur_key = f"career_history_{i}_duration_months"
        if dur_key in record:
            try:
                total_career_months += float(record[dur_key])
            except (TypeError, ValueError):
                pass
        else:
            if i > 5 and not any(k.startswith(f"career_history_{i}") for k in record):
                break
                
    if candidate_experience is not None and candidate_experience > 0:
        expected_months = candidate_experience * 12
        if total_career_months > 0 and total_career_months < expected_months * 0.4:
            return True
            
    return False


def behavioral_score(record: dict[str, Any]) -> float:
    """Aggregate recruiter-signal fields into a bounded numeric score."""
    from datetime import datetime, timezone
    score_components: list[float] = []
    boolean_bonus = 0.0
    
    for key, value in record.items():
        key_text = str(key)
        if not key_text.startswith("redrob_signals_"):
            continue
            
        if "expected_salary_range" in key_text:
            continue
            
        if isinstance(value, bool):
            boolean_bonus += 1.0 if value else 0.0
            continue
            
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
            
        if "notice_period_days" in key_text:
            comp = max(0.0, 1.0 - (val / 90.0))
            score_components.append(comp)
        elif "avg_response_time_hours" in key_text:
            comp = max(0.0, 1.0 - (val / 48.0))
            score_components.append(comp)
        else:
            comp = min(max(val, 0.0), 100.0) / 100.0
            score_components.append(comp)
            
    if not score_components and boolean_bonus == 0.0:
        base_score = 0.0
    else:
        base_score = sum(score_components) / max(len(score_components), 1)
        base_score = min(1.0, base_score + (0.1 * boolean_bonus))
        
    last_active = record.get("redrob_signals_last_active_date") or record.get("last_active_date")
    if last_active:
        try:
            dt = datetime.strptime(str(last_active)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_ago = (datetime.now(timezone.utc) - dt).days
            if days_ago > 180:
                base_score *= 0.5
        except ValueError:
            pass
            
    return float(base_score)


def reciprocal_rank_fusion(
    *rank_lists: list[str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists of candidate IDs using Reciprocal Rank Fusion.

    RRF is a robust, parameter-light method for combining rankings from
    heterogeneous retrieval systems (e.g., embedding-based + BM25).

    Parameters
    ----------
    *rank_lists:
        Each list is a ranking of candidate_id strings, best-first.
    k:
        Smoothing constant (typically 60, per Cormack et al. 2009).

    Returns
    -------
    List of (candidate_id, rrf_score) tuples, sorted descending by score.
    """
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for position, candidate_id in enumerate(rank_list, start=1):
            scores[candidate_id] = scores.get(candidate_id, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def first_present(record: dict[str, Any], keys: Iterable[str]) -> str:
    """Return the first meaningful text value for a set of keys.

    Returns an empty string (not a fake title) when no value is found,
    so that title_match_score() correctly returns 0.0 instead of matching
    the literal string ``"Candidate"``.
    """
    for key in keys:
        value = record.get(key)
        if is_present(value) and str(value).strip():
            return str(value)
    return ""


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
