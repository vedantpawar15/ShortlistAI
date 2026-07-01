"""Explanation generation for recruiter-facing ranking decisions.

Design notes
------------
* ``RankingExplainer.explain()`` produces a single structured explanation.
* ``RankingExplainer.explain_batch()`` accepts the full ranked DataFrame and
  annotates each row with percentile context (e.g. "top 12%").
* Score thresholds are derived from batch percentiles, not hardcoded constants.
* ``score_breakdown`` now includes all seven feature dimensions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class CandidateExplanation(BaseModel):
    """Structured explanation for a candidate ranking result."""

    candidate_id: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    percentile_rank: float | None = Field(default=None, description="Percentile rank in [0, 100] (100 = best).")


class RankingExplainer:
    """Create transparent explanations for ranked candidates."""

    # Feature keys included in the score_breakdown output.
    SCORE_FEATURES = (
        "semantic_similarity",
        "lexical_similarity",
        "skill_overlap",
        "experience_match",
        "education_match",
        "title_similarity",
        "location_match",
        "behavioral_signal_score",
    )

    # Human-readable labels for each feature.
    FEATURE_LABELS: dict[str, str] = {
        "semantic_similarity": "Semantic similarity (embedding)",
        "lexical_similarity": "Lexical text overlap",
        "bm25_similarity": "BM25 relevance score",
        "skill_overlap": "Skill coverage",
        "experience_match": "Experience level match",
        "education_match": "Education relevance",
        "title_similarity": "Job title alignment",
        "location_match": "Location match",
        "behavioral_signal_score": "Behavioral signals",
    }

    def explain(
        self,
        candidate_id: str,
        features: dict[str, Any],
        *,
        percentile_rank: float | None = None,
        high_threshold: float = 0.7,
        low_threshold: float = 0.4,
    ) -> CandidateExplanation:
        """Generate a structured explanation for one candidate.

        Parameters
        ----------
        candidate_id:
            Identifier of the candidate being explained.
        features:
            Feature dictionary (row from the scored DataFrame).
        percentile_rank:
            Pre-computed percentile (0–100, 100 = best).  Pass ``None``
            to omit percentile context from the summary.
        high_threshold, low_threshold:
            Thresholds for labelling features as strengths or gaps.
            Derived from batch statistics when ``explain_batch()`` is used.
        """
        score_breakdown: dict[str, float] = {}
        for key in self.SCORE_FEATURES:
            raw = features.get(key)
            if raw is not None:
                try:
                    score_breakdown[key] = round(float(raw), 4)
                except (TypeError, ValueError):
                    pass

        # Also capture BM25 if present.
        bm25 = features.get("bm25_similarity")
        if bm25 is not None:
            try:
                score_breakdown["bm25_similarity"] = round(float(bm25), 4)
            except (TypeError, ValueError):
                pass

        strengths: list[str] = []
        gaps: list[str] = []

        # Skill signals.
        matched_skills = features.get("matched_skills", [])
        missing_skills = features.get("missing_skills", [])
        if matched_skills:
            skills_preview = ", ".join(str(s) for s in matched_skills[:5])
            strengths.append(f"Strong skill match: {skills_preview}")
        if missing_skills:
            priority_gaps = ", ".join(str(s) for s in missing_skills[:5])
            gaps.append(f"Missing required skills: {priority_gaps}")

        # Feature-level strengths and gaps.
        feature_gap_messages = {
            "semantic_similarity": (
                "Strong semantic alignment with the job description",
                "Low semantic alignment with the job description",
            ),
            "experience_match": (
                "Experience level meets or exceeds the role requirement",
                "Experience signal is below the target requirement",
            ),
            "title_similarity": (
                "Current title closely aligns with the target role",
                None,  # Not a critical gap.
            ),
            "location_match": (
                "Located in or near the target location",
                None,
            ),
            "behavioral_signal_score": (
                "Strong positive behavioral signals (open-to-work, GitHub activity, etc.)",
                None,
            ),
            "education_match": (
                "Educational background is relevant to the role",
                "Education relevance to the role is limited",
            ),
        }

        for feature_key, (strength_msg, gap_msg) in feature_gap_messages.items():
            value = score_breakdown.get(feature_key)
            if value is None:
                continue
            if value >= high_threshold and strength_msg:
                strengths.append(strength_msg)
            elif value < low_threshold and gap_msg:
                gaps.append(gap_msg)

        # Build the summary string.
        if features.get("is_honeypot"):
            summary = "Candidate disqualified due to inconsistent profile information (honeypot flags)."
        else:
            strength_text = f"Strong fit: {strengths[0]}" if strengths else "Meets core requirements."
            gap_text = f"Gap: {gaps[0]}" if gaps else "No major gaps."
            
            beh_score = features.get("behavioral_signal_score")
            beh_text = ""
            if beh_score is not None:
                try:
                    beh_score_val = float(beh_score)
                    if beh_score_val < 0.4:
                        beh_text = " Candidate has weak behavioral signals."
                    elif beh_score_val > 0.8:
                        beh_text = " Candidate has highly positive behavioral signals."
                except (ValueError, TypeError):
                    pass

            percentile_text = ""
            if percentile_rank is not None:
                percentile_text = f" (Top {100 - percentile_rank:.0f}%)"

            summary = f"{strength_text} {gap_text}{beh_text}{percentile_text}"


        return CandidateExplanation(
            candidate_id=str(candidate_id),
            summary=summary,
            strengths=strengths,
            gaps=gaps,
            score_breakdown=score_breakdown,
            percentile_rank=percentile_rank,
        )

    def explain_batch(
        self,
        ranked: pd.DataFrame,
    ) -> list[CandidateExplanation]:
        """Generate explanations for all candidates in a ranked DataFrame.

        Thresholds for strengths/gaps are derived from the batch distribution
        (75th / 25th percentile) so that explanations are relative to this
        specific applicant pool rather than hardcoded constants.

        Returns a list aligned with the input DataFrame row order.
        """
        if ranked.empty:
            return []

        scores = ranked["score"].astype(float) if "score" in ranked.columns else pd.Series([0.5] * len(ranked))
        high_threshold = float(scores.quantile(0.75)) if len(scores) > 1 else 0.7
        low_threshold = float(scores.quantile(0.25)) if len(scores) > 1 else 0.4

        # Clamp thresholds to prevent degenerate cases.
        high_threshold = max(high_threshold, 0.5)
        low_threshold = min(low_threshold, 0.5)

        n = len(ranked)
        explanations: list[CandidateExplanation] = []
        for rank_position, (_, row) in enumerate(ranked.iterrows(), start=1):
            percentile_rank = round((1 - rank_position / n) * 100, 1) if n > 0 else None
            explanations.append(
                self.explain(
                    str(row.get("candidate_id", "")),
                    row.to_dict(),
                    percentile_rank=percentile_rank,
                    high_threshold=high_threshold,
                    low_threshold=low_threshold,
                )
            )
        return explanations
