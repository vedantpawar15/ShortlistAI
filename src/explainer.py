"""Explanation generation for recruiter-facing ranking decisions."""

from __future__ import annotations

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

    def explain(self, candidate_id: str, features: dict[str, float]) -> CandidateExplanation:
        """Generate an explanation for one candidate."""
        raise NotImplementedError("Ranking explanations will be implemented later.")

