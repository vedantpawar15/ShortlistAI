"""Explanation generation for recruiter-facing ranking decisions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CandidateExplanation(BaseModel):
    """Structured explanation for a candidate ranking result."""

    candidate_id: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class RankingExplainer:
    """Create transparent explanations for ranked candidates."""

    def explain(self, candidate_id: str, features: dict[str, Any]) -> CandidateExplanation:
        """Generate an explanation for one candidate."""
        score_breakdown = {
            key: round(float(features.get(key, 0.0)), 4)
            for key in ("semantic_similarity", "skill_overlap", "experience_match", "education_match")
        }
        strengths: list[str] = []
        gaps: list[str] = []

        matched_skills = features.get("matched_skills", [])
        missing_skills = features.get("missing_skills", [])
        if matched_skills:
            strengths.append(f"Matches skills: {', '.join(str(skill) for skill in matched_skills[:5])}")
        if score_breakdown["semantic_similarity"] >= 0.5:
            strengths.append("Strong textual match to the job description")
        if score_breakdown["experience_match"] >= 0.8:
            strengths.append("Experience level meets the role requirement")
        if missing_skills:
            gaps.append(f"Missing explicit skills: {', '.join(str(skill) for skill in missing_skills[:5])}")
        if score_breakdown["experience_match"] < 0.5:
            gaps.append("Experience signal is below the target requirement")

        summary = (
            f"Overall fit is driven by semantic similarity {score_breakdown['semantic_similarity']:.2f}, "
            f"skill overlap {score_breakdown['skill_overlap']:.2f}, and experience match "
            f"{score_breakdown['experience_match']:.2f}."
        )
        return CandidateExplanation(
            candidate_id=str(candidate_id),
            summary=summary,
            strengths=strengths,
            gaps=gaps,
            score_breakdown=score_breakdown,
        )

