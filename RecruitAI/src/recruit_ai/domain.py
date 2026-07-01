"""Domain models for candidate ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateProfile:
    candidate_id: str
    name: str
    headline: str
    summary: str
    skills: list[str]
    years_experience: float
    location: str
    desired_titles: list[str] = field(default_factory=list)
    work_experiences: list[str] = field(default_factory=list)
    achievements: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    updated_days_ago: int = 30

    def searchable_text(self) -> str:
        return " ".join(
            [
                self.name,
                self.headline,
                self.summary,
                " ".join(self.skills),
                " ".join(self.desired_titles),
                " ".join(self.work_experiences),
                " ".join(self.achievements),
                " ".join(self.education),
                " ".join(self.certifications),
                self.location,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobRequirement:
    job_id: str
    title: str
    summary: str
    required_skills: list[str]
    preferred_skills: list[str] = field(default_factory=list)
    minimum_years_experience: float = 0.0
    location: str = "Remote"
    responsibilities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def searchable_text(self) -> str:
        return " ".join(
            [
                self.title,
                self.summary,
                " ".join(self.required_skills),
                " ".join(self.preferred_skills),
                " ".join(self.responsibilities),
                " ".join(self.keywords),
                self.location,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankingResult:
    candidate: CandidateProfile
    total_score: float
    feature_scores: dict[str, float]
    feature_contributions: dict[str, float]
    matched_required_skills: list[str]
    matched_preferred_skills: list[str]
    missing_required_skills: list[str]
    explanation: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate"] = self.candidate.to_dict()
        return payload
