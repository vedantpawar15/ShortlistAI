"""Candidate text preprocessing for semantic ranking profiles."""

from __future__ import annotations

import json
import re
import string
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.logging_utils import logger

try:
    from rapidfuzz import fuzz, process
except ImportError:
    from difflib import SequenceMatcher

    class fuzz:
        """Fallback subset of rapidfuzz.fuzz."""

        @staticmethod
        def WRatio(left: object, right: object) -> float:
            return SequenceMatcher(None, str(left), str(right)).ratio() * 100

    class process:
        """Fallback subset of rapidfuzz.process."""

        @staticmethod
        def extractOne(query: object, choices: Iterable[object], scorer: object) -> tuple[object, float, int] | None:
            best: tuple[object, float, int] | None = None
            for index, choice in enumerate(choices):
                score = scorer(query, choice)
                if best is None or score > best[1]:
                    best = (choice, score, index)
            return best

SECTION_ORDER = (
    "summary",
    "headline",
    "career_history",
    "skills",
    "education",
    "certifications",
    "languages",
    "behavior_signals",
    "experience",
    "current_role",
    "current_company",
)

JOB_SECTION_ORDER = (
    "title",
    "summary",
    "required_skills",
    "preferred_skills",
    "experience",
    "education",
    "location",
)

CAREER_HISTORY_FIELD_ORDER = (
    "title",
    "role",
    "position",
    "company",
    "organization",
    "industry",
    "company_size",
    "description",
    "start_date",
    "end_date",
    "duration_months",
    "is_current",
)

DEFAULT_SKILL_ALIASES = {
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "js": "javascript",
    "javascript": "javascript",
    "k8": "kubernetes",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "ml": "machine learning",
    "machine learning": "machine learning",
    "nlp": "natural language processing",
    "natural language processing": "natural language processing",
    "py": "python",
    "python": "python",
    "tf": "tensorflow",
    "tensorflow": "tensorflow",
}


@dataclass(frozen=True)
class SemanticProfile:
    """Structured semantic profile generated for one candidate."""

    candidate_id: str | None
    document: str
    sections: dict[str, str] = field(default_factory=dict)
    normalized_skills: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class JobProfile:
    """Structured semantic profile generated for one job."""

    job_id: str | None
    document: str
    sections: dict[str, str] = field(default_factory=dict)
    normalized_skills: list[str] = field(default_factory=list)


class SkillNormalizer:
    """Normalize skill names with aliases and fuzzy matching."""

    def __init__(self, aliases: Mapping[str, str] | None = None, fuzzy_threshold: int = 88) -> None:
        self.aliases = dict(DEFAULT_SKILL_ALIASES)
        if aliases:
            self.aliases.update({self._clean_key(key): self._clean_key(value) for key, value in aliases.items()})
        self.fuzzy_threshold = fuzzy_threshold
        self.vocabulary = sorted(set(self.aliases.values()) | set(self.aliases.keys()))

    def normalize(self, skill: object) -> str:
        """Return a canonical skill name when a confident mapping exists."""
        cleaned = self._clean_key(skill)
        if not cleaned:
            return ""
        if cleaned in self.aliases:
            return self.aliases[cleaned]

        match = process.extractOne(cleaned, self.vocabulary, scorer=fuzz.WRatio)
        if match and match[1] >= self.fuzzy_threshold:
            matched_value = match[0]
            return self.aliases.get(matched_value, matched_value)
        return cleaned

    def normalize_many(self, skills: Iterable[object]) -> list[str]:
        """Normalize, de-duplicate, and sort skills in input order."""
        normalized: list[str] = []
        seen: set[str] = set()
        for skill in skills:
            value = self.normalize(skill)
            if value and value not in seen:
                normalized.append(value)
                seen.add(value)
        return normalized

    @staticmethod
    def _clean_key(value: object) -> str:
        text = "" if value is None else str(value)
        text = unicodedata.normalize("NFKC", text)
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text


class TextPreprocessor:
    """Build cleaned semantic candidate documents for ranking."""

    whitespace_pattern = re.compile(r"\s+")
    punctuation_translation = str.maketrans({character: " " for character in string.punctuation})

    def __init__(self, skill_normalizer: SkillNormalizer | None = None) -> None:
        self.skill_normalizer = skill_normalizer or SkillNormalizer()

    def preprocess_candidates(self, candidates: pd.DataFrame | Iterable[Mapping[str, Any]]) -> list[str]:
        """Convert every candidate row into one cleaned semantic document."""
        return [profile.document for profile in self.build_candidate_profiles(candidates)]

    def preprocess_jobs(self, jobs: pd.DataFrame | Iterable[Mapping[str, Any]]) -> list[str]:
        """Convert every job row into one cleaned semantic document."""
        return [profile.document for profile in self.build_job_profiles(jobs)]

    def build_candidate_profiles(self, candidates: pd.DataFrame | Iterable[Mapping[str, Any]]) -> list[SemanticProfile]:
        """Return structured semantic profiles for all candidates."""
        records = self._records_from_candidates(candidates)
        prepared_records: list[dict[str, Any]] = []
        career_key_pattern = re.compile(r"career_history_(\d+)_(.+)")

        for record in records:
            prepared = dict(record)
            if not isinstance(prepared.get("career_history"), list):
                grouped_roles: dict[int, dict[str, Any]] = {}
                for key, value in record.items():
                    match = career_key_pattern.fullmatch(key)
                    if match and is_meaningful(value):
                        index = int(match.group(1))
                        field_name = match.group(2)
                        grouped_roles.setdefault(index, {})[field_name] = value

                if grouped_roles:
                    prepared["career_history"] = [grouped_roles[index] for index in sorted(grouped_roles)]
                    for key in list(prepared):
                        if career_key_pattern.fullmatch(key):
                            prepared.pop(key)

            prepared_records.append(prepared)

        profiles = [self.build_candidate_profile(record) for record in prepared_records]
        logger.info("Built {} semantic candidate profiles", len(profiles))
        return profiles

    def build_job_profiles(self, jobs: pd.DataFrame | Iterable[Mapping[str, Any]]) -> list[JobProfile]:
        """Return structured semantic profiles for all jobs."""
        records = self._records_from_candidates(jobs)
        profiles = [self.build_job_profile(record) for record in records]
        logger.info("Built {} semantic job profiles", len(profiles))
        return profiles

    def build_candidate_profile(self, candidate: Mapping[str, Any] | pd.Series) -> SemanticProfile:
        """Build one structured semantic profile from a candidate record."""
        record = self._record_to_dict(candidate)
        normalized_skills = self.extract_normalized_skills(record)
        sections = {
            "summary": self._collect_summary(record),
            "headline": self._collect_headline(record),
            "career_history": self._collect_career_history(record),
            "skills": " ".join(normalized_skills),
            "education": self._collect_education(record),
            "certifications": self._collect_certifications(record),
            "languages": self._collect_languages(record),
            "behavior_signals": self._collect_behavior_signals(record),
            "experience": self._collect_experience(record),
            "current_role": self._collect_current_role(record),
            "current_company": self._collect_current_company(record),
        }
        cleaned_sections = {
            section: self.preprocess(text)
            for section, text in sections.items()
            if self.preprocess(text)
        }
        document_parts = [
            f"{section} {cleaned_sections[section]}"
            for section in SECTION_ORDER
            if section in cleaned_sections
        ]
        document = self.preprocess(" ".join(document_parts))
        candidate_id = self._first_value(record, "candidate_id", "id")
        return SemanticProfile(
            candidate_id=candidate_id,
            document=document,
            sections=cleaned_sections,
            normalized_skills=normalized_skills,
        )

    def build_job_profile(self, job: Mapping[str, Any] | pd.Series) -> JobProfile:
        """Build one structured semantic profile from a job record."""
        record = self._record_to_dict(job)
        normalized_skills = self.extract_normalized_job_skills(record)
        sections = {
            "title": self._collect_job_title(record),
            "summary": self._collect_job_summary(record),
            "required_skills": self._collect_job_skills(record, ("required_skills",)),
            "preferred_skills": self._collect_job_skills(record, ("preferred_skills",)),
            "experience": self._collect_job_experience(record),
            "education": self._collect_job_education(record),
            "location": self._collect_job_location(record),
        }
        cleaned_sections = {
            section: self.preprocess(text)
            for section, text in sections.items()
            if self.preprocess(text)
        }
        if normalized_skills:
            cleaned_sections["normalized_skills"] = self.preprocess(" ".join(normalized_skills))
        document_parts = [
            f"{section} {cleaned_sections[section]}"
            for section in JOB_SECTION_ORDER
            if section in cleaned_sections
        ]
        if "normalized_skills" in cleaned_sections:
            document_parts.append(f"skills {cleaned_sections['normalized_skills']}")
        document = self.preprocess(" ".join(document_parts))
        job_id = self._first_value(record, "job_id", "id")
        return JobProfile(
            job_id=job_id,
            document=document,
            sections=cleaned_sections,
            normalized_skills=normalized_skills,
        )

    def extract_normalized_skills(self, candidate: Mapping[str, Any] | pd.Series) -> list[str]:
        """Extract and normalize skills from raw or flattened candidate records."""
        record = self._record_to_dict(candidate)
        skills: list[object] = []

        raw_skills = record.get("skills")
        if isinstance(raw_skills, list):
            skills.extend(self._extract_names_from_objects(raw_skills, preferred_keys=("name", "skill")))
        elif raw_skills:
            skills.extend(split_skill_text(raw_skills))

        skills.extend(value for key, value in record.items() if re.fullmatch(r"skills_\d+_name", key) and value)
        skills.extend(
            value
            for key, value in record.items()
            if re.fullmatch(r"redrob_signals_skill_assessment_scores_[^_]+", key) and value is not None
        )
        return self.skill_normalizer.normalize_many(skills)

    def extract_normalized_job_skills(self, job: Mapping[str, Any] | pd.Series) -> list[str]:
        """Extract and normalize skills from raw job fields and description text."""
        record = self._record_to_dict(job)
        skills: list[object] = []
        for field_name in ("required_skills", "preferred_skills", "skills"):
            raw_value = record.get(field_name)
            if isinstance(raw_value, list):
                skills.extend(self._extract_names_from_objects(raw_value, preferred_keys=("name", "skill")))
            elif raw_value:
                skills.extend(split_skill_text(raw_value))

        for field_name in ("title", "description", "summary"):
            value = record.get(field_name)
            if value:
                skills.extend(extract_alias_matches(str(value)))
        return self.skill_normalizer.normalize_many(skills)

    def clean_text(self, text: object) -> str:
        """Normalize unicode, case, punctuation, and whitespace."""
        normalized = self.normalize_unicode(text)
        normalized = self.normalize_case(normalized)
        normalized = self.normalize_punctuation(normalized)
        return self.normalize_whitespace(normalized)

    def normalize_unicode(self, text: object) -> str:
        """Apply compatibility unicode normalization and ASCII-safe transliteration."""
        value = "" if text is None else str(text)
        value = unicodedata.normalize("NFKD", value)
        value = "".join(" " if unicodedata.category(character).startswith("P") else character for character in value)
        return value.encode("ascii", "ignore").decode("ascii")

    def normalize_case(self, text: str) -> str:
        """Normalize text casing for lexical and semantic consistency."""
        return text.lower()

    def normalize_punctuation(self, text: str) -> str:
        """Replace punctuation with spaces to avoid token collisions."""
        return text.translate(self.punctuation_translation)

    def normalize_whitespace(self, text: str) -> str:
        """Collapse repeated whitespace and trim edges."""
        return self.whitespace_pattern.sub(" ", text).strip()

    def preprocess(self, text: object) -> str:
        """Apply the full text normalization pipeline."""
        return self.clean_text(text)

    def _collect_summary(self, record: Mapping[str, Any]) -> str:
        parts = self._values_for_keys(record, ("profile_summary", "summary", "resume_text"))
        profile = record.get("profile")
        if isinstance(profile, Mapping):
            parts.extend(self._values_for_keys(profile, ("summary",)))
        return join_values(parts)

    def _collect_headline(self, record: Mapping[str, Any]) -> str:
        keys = ("profile_headline", "headline", "profile_current_title", "current_title", "profile_current_industry")
        parts = self._values_for_keys(record, keys)
        profile = record.get("profile")
        if isinstance(profile, Mapping):
            parts.extend(self._values_for_keys(profile, ("headline", "current_title", "current_industry")))
        return join_values(parts)

    def _collect_current_role(self, record: Mapping[str, Any]) -> str:
        parts = self._values_for_keys(record, ("profile_current_title", "current_title", "current_role", "title"))
        profile = record.get("profile")
        if isinstance(profile, Mapping):
            parts.extend(self._values_for_keys(profile, ("current_title", "current_role", "title")))
        return join_values(parts)

    def _collect_current_company(self, record: Mapping[str, Any]) -> str:
        parts = self._values_for_keys(record, ("profile_current_company", "current_company", "company"))
        profile = record.get("profile")
        if isinstance(profile, Mapping):
            parts.extend(self._values_for_keys(profile, ("current_company", "company")))
        return join_values(parts)

    def _collect_experience(self, record: Mapping[str, Any]) -> str:
        parts = self._values_for_keys(
            record,
            (
                "profile_years_of_experience",
                "years_of_experience",
                "experience_years",
                "total_experience_years",
                "experience",
            ),
        )
        profile = record.get("profile")
        if isinstance(profile, Mapping):
            parts.extend(
                self._values_for_keys(
                    profile,
                    ("years_of_experience", "experience_years", "total_experience_years", "experience"),
                )
            )
        return join_values(parts)

    def _collect_career_history(self, record: Mapping[str, Any]) -> str:
        parts: list[object] = []
        career_history = record.get("career_history")
        if isinstance(career_history, list):
            for role in career_history:
                if isinstance(role, Mapping):
                    parts.extend(ordered_mapping_values(role, CAREER_HISTORY_FIELD_ORDER))
        parts.extend(
            value
            for key, value in sorted(record.items(), key=career_history_sort_key)
            if re.fullmatch(r"career_history_\d+_.+", key) and is_meaningful(value)
        )
        return join_values(parts)

    def _collect_education(self, record: Mapping[str, Any]) -> str:
        parts: list[object] = []
        education = record.get("education")
        if isinstance(education, list):
            for item in education:
                if isinstance(item, Mapping):
                    parts.extend(
                        self._values_for_keys(
                            item,
                            ("institution", "degree", "field_of_study", "grade", "tier"),
                        )
                    )
        parts.extend(
            value
            for key, value in sorted(record.items())
            if re.fullmatch(r"education_\d+_(institution|degree|field_of_study|grade|tier)", key) and value
        )
        return join_values(parts)

    def _collect_certifications(self, record: Mapping[str, Any]) -> str:
        parts: list[object] = []
        certifications = record.get("certifications")
        if isinstance(certifications, list):
            for item in certifications:
                if isinstance(item, Mapping):
                    parts.extend(self._values_for_keys(item, ("name", "issuer", "year")))
        parts.extend(
            value
            for key, value in sorted(record.items())
            if re.fullmatch(r"certifications_\d+_(name|issuer|year)", key) and value
        )
        return join_values(parts)

    def _collect_languages(self, record: Mapping[str, Any]) -> str:
        parts: list[object] = []
        languages = record.get("languages")
        if isinstance(languages, list):
            for item in languages:
                if isinstance(item, Mapping):
                    parts.extend(self._values_for_keys(item, ("language", "proficiency")))
        parts.extend(
            value
            for key, value in sorted(record.items())
            if re.fullmatch(r"languages_\d+_(language|proficiency)", key) and value
        )
        return join_values(parts)

    def _collect_behavior_signals(self, record: Mapping[str, Any]) -> str:
        parts: list[str] = []
        signals = record.get("redrob_signals")
        if isinstance(signals, Mapping):
            parts.extend(format_signal(key, value) for key, value in sorted(signals.items()) if is_meaningful(value))
        parts.extend(
            format_signal(key.removeprefix("redrob_signals_"), value)
            for key, value in sorted(record.items())
            if key.startswith("redrob_signals_") and is_meaningful(value)
        )
        return join_values(parts)

    def _collect_job_title(self, record: Mapping[str, Any]) -> str:
        return join_values(self._values_for_keys(record, ("title", "job_title")))

    def _collect_job_summary(self, record: Mapping[str, Any]) -> str:
        return join_values(self._values_for_keys(record, ("summary", "description", "job_description")))

    def _collect_job_skills(self, record: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        parts: list[object] = []
        for key in keys:
            raw_value = record.get(key)
            if isinstance(raw_value, list):
                parts.extend(self._extract_names_from_objects(raw_value, preferred_keys=("name", "skill")))
            elif raw_value:
                parts.extend(split_skill_text(raw_value))
        return join_values(parts)

    def _collect_job_experience(self, record: Mapping[str, Any]) -> str:
        return join_values(
            self._values_for_keys(
                record,
                ("experience_years", "years_of_experience", "required_experience_years", "experience"),
            )
        )

    def _collect_job_education(self, record: Mapping[str, Any]) -> str:
        return join_values(self._values_for_keys(record, ("education", "required_education", "preferred_education")))

    def _collect_job_location(self, record: Mapping[str, Any]) -> str:
        return join_values(self._values_for_keys(record, ("location", "job_location")))

    def _records_from_candidates(self, candidates: pd.DataFrame | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(candidates, pd.DataFrame):
            return candidates.to_dict(orient="records")
        return [self._record_to_dict(candidate) for candidate in candidates]

    def _record_to_dict(self, candidate: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
        if isinstance(candidate, pd.Series):
            return candidate.dropna().to_dict()
        return dict(candidate)

    def _first_value(self, record: Mapping[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = record.get(key)
            if is_meaningful(value):
                return str(value)
        return None

    def _values_for_keys(self, record: Mapping[str, Any], keys: Iterable[str]) -> list[object]:
        return [record[key] for key in keys if key in record and is_meaningful(record[key])]

    def _extract_names_from_objects(self, values: Iterable[Any], preferred_keys: tuple[str, ...]) -> list[object]:
        extracted: list[object] = []
        for value in values:
            if isinstance(value, Mapping):
                extracted.extend(self._values_for_keys(value, preferred_keys))
            else:
                extracted.append(value)
        return extracted


def split_skill_text(value: object) -> list[str]:
    """Split serialized skill text into individual skill tokens."""
    if value is None:
        return []
    if isinstance(value, str):
        parsed = parse_json_if_possible(value)
        if isinstance(parsed, list):
            return [str(item.get("name", item)) if isinstance(item, Mapping) else str(item) for item in parsed]
        return [part.strip() for part in re.split(r"\s*(?:\||,|;|/)\s*", value) if part.strip()]
    return [str(value)]


def extract_alias_matches(text: str) -> list[str]:
    """Extract known skill aliases from free-form text."""
    lowered = text.lower()
    matches: list[str] = []
    for alias in DEFAULT_SKILL_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            matches.append(alias)
    return matches


def parse_json_if_possible(value: str) -> Any:
    """Parse JSON strings produced by flattened nested lists when possible."""
    stripped = value.strip()
    if not stripped.startswith(("[", "{")):
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def join_values(values: Iterable[object]) -> str:
    """Join meaningful values into one text block."""
    return " ".join(str(value) for value in values if is_meaningful(value))


def is_meaningful(value: object) -> bool:
    """Return True when a value should contribute semantic text."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, dict)) and not value:
        return False
    return True


def format_signal(key: object, value: object) -> str:
    """Format behavior signal key-value pairs as readable semantic text."""
    key_text = str(key).replace("_", " ")
    if isinstance(value, Mapping):
        nested = " ".join(
            format_signal(nested_key, nested_value)
            for nested_key, nested_value in value.items()
            if is_meaningful(nested_value)
        )
        return f"{key_text} {nested}"
    if isinstance(value, list):
        return f"{key_text} {join_values(value)}"
    return f"{key_text} {value}"


def ordered_mapping_values(record: Mapping[str, Any], preferred_keys: Iterable[str]) -> list[object]:
    """Return meaningful mapping values in semantic order, then append remaining fields."""
    values: list[object] = []
    seen: set[str] = set()
    for key in preferred_keys:
        if key in record and is_meaningful(record[key]):
            values.append(record[key])
            seen.add(key)
    for key, value in record.items():
        if key not in seen and is_meaningful(value):
            values.append(value)
    return values


def career_history_sort_key(item: tuple[str, Any]) -> tuple[int, int, str]:
    """Sort flattened career-history fields by role index and semantic field order."""
    key, _ = item
    match = re.fullmatch(r"career_history_(\d+)_(.+)", key)
    if not match:
        return (10**9, len(CAREER_HISTORY_FIELD_ORDER), key)
    index = int(match.group(1))
    field_name = match.group(2)
    try:
        field_order = CAREER_HISTORY_FIELD_ORDER.index(field_name)
    except ValueError:
        field_order = len(CAREER_HISTORY_FIELD_ORDER)
    return (index, field_order, field_name)
