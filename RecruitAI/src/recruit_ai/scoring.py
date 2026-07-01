"""Hybrid ranking logic with explicit feature contributions."""

from __future__ import annotations

import math
from collections import Counter

from .config import ScoringConfig
from .domain import CandidateProfile, JobRequirement, RankingResult
from .nlp import SearchHit, TfidfIndex, reciprocal_rank_fusion, skill_key, tokenize


def _bounded_ratio(value: float, target: float) -> float:
    if target <= 0:
        return 1.0
    if value >= target:
        extra = min(value - target, target * 2)
        return min(1.0, 0.85 + 0.15 * (extra / max(target * 2, 1e-6)))
    return max(0.0, value / target)


def _location_score(candidate_location: str, job_location: str) -> float:
    candidate_norm = candidate_location.lower()
    job_norm = job_location.lower()
    if "remote" in candidate_norm and "remote" in job_norm:
        return 1.0
    if "remote" in job_norm:
        return 0.9
    if candidate_norm == job_norm:
        return 1.0
    if any(part and part in candidate_norm for part in job_norm.replace(",", " ").split()):
        return 0.75
    return 0.2


def _title_alignment(candidate: CandidateProfile, job: JobRequirement) -> float:
    title_terms = set(tokenize(job.title))
    candidate_terms = set(tokenize(candidate.headline + " " + " ".join(candidate.desired_titles)))
    if not title_terms:
        return 0.0
    return len(title_terms.intersection(candidate_terms)) / len(title_terms)


def _achievement_intensity(candidate: CandidateProfile) -> float:
    quantified = 0
    for achievement in candidate.achievements:
        if any(char.isdigit() for char in achievement):
            quantified += 1
    if not candidate.achievements:
        return 0.0
    return quantified / len(candidate.achievements)


class HybridRanker:
    def __init__(self, config: ScoringConfig) -> None:
        self.config = config

    def rank(self, job: JobRequirement, candidates: list[CandidateProfile]) -> list[RankingResult]:
        documents = {candidate.candidate_id: candidate.searchable_text() for candidate in candidates}
        index = TfidfIndex(documents)
        semantic_scores = index.semantic_scores(job.searchable_text())
        bm25_scores = index.bm25_scores(
            job.searchable_text(),
            k1=self.config.search.bm25_k1,
            b=self.config.search.bm25_b,
        )

        semantic_ranking = [SearchHit(document_id=doc_id, score=score) for doc_id, score in semantic_scores.items()]
        bm25_ranking = [SearchHit(document_id=doc_id, score=score) for doc_id, score in bm25_scores.items()]
        fused_scores = reciprocal_rank_fusion([semantic_ranking, bm25_ranking])
        max_bm25 = max(bm25_scores.values(), default=1.0) or 1.0

        results: list[RankingResult] = []
        for candidate in candidates:
            result = self._score_candidate(
                candidate=candidate,
                job=job,
                semantic_score=semantic_scores.get(candidate.candidate_id, 0.0),
                bm25_score=bm25_scores.get(candidate.candidate_id, 0.0) / max_bm25,
                fusion_score=fused_scores.get(candidate.candidate_id, 0.0),
            )
            results.append(result)
        return sorted(results, key=lambda item: item.total_score, reverse=True)

    def _score_candidate(
        self,
        candidate: CandidateProfile,
        job: JobRequirement,
        semantic_score: float,
        bm25_score: float,
        fusion_score: float,
    ) -> RankingResult:
        weights = self.config.feature_weights
        required = {skill_key(skill): skill for skill in job.required_skills}
        preferred = {skill_key(skill): skill for skill in job.preferred_skills}
        candidate_skills = {skill_key(skill): skill for skill in candidate.skills}

        matched_required_keys = sorted(set(required).intersection(candidate_skills))
        matched_preferred_keys = sorted(set(preferred).intersection(candidate_skills))
        missing_required_keys = sorted(set(required).difference(candidate_skills))

        required_overlap = len(matched_required_keys) / max(len(required), 1)
        preferred_overlap = len(matched_preferred_keys) / max(len(preferred), 1) if preferred else 1.0
        experience_fit = _bounded_ratio(candidate.years_experience, job.minimum_years_experience)
        title_alignment = _title_alignment(candidate, job)
        location_alignment = _location_score(candidate.location, job.location)
        recency = math.exp(-candidate.updated_days_ago / 120.0)
        achievement_intensity = _achievement_intensity(candidate)

        feature_scores = {
            "semantic_similarity": max(semantic_score, self.config.search.semantic_floor if required_overlap > 0 else 0.0),
            "bm25_relevance": max(0.0, bm25_score),
            "required_skill_overlap": required_overlap,
            "preferred_skill_overlap": preferred_overlap,
            "experience_fit": experience_fit,
            "title_alignment": title_alignment,
            "location_alignment": location_alignment,
            "recency": recency,
            "achievement_intensity": achievement_intensity,
        }
        feature_contributions = {
            feature: getattr(weights, feature) * score for feature, score in feature_scores.items()
        }
        total_score = sum(feature_contributions.values()) + (0.03 * fusion_score)
        total_score = min(total_score, 1.0)

        explanation = self._explanation(
            candidate=candidate,
            matched_required=[required[key] for key in matched_required_keys],
            matched_preferred=[preferred[key] for key in matched_preferred_keys],
            missing_required=[required[key] for key in missing_required_keys],
            feature_contributions=feature_contributions,
        )
        return RankingResult(
            candidate=candidate,
            total_score=round(total_score, 4),
            feature_scores={key: round(value, 4) for key, value in feature_scores.items()},
            feature_contributions={key: round(value, 4) for key, value in feature_contributions.items()},
            matched_required_skills=[required[key] for key in matched_required_keys],
            matched_preferred_skills=[preferred[key] for key in matched_preferred_keys],
            missing_required_skills=[required[key] for key in missing_required_keys],
            explanation=explanation,
        )

    def _explanation(
        self,
        candidate: CandidateProfile,
        matched_required: list[str],
        matched_preferred: list[str],
        missing_required: list[str],
        feature_contributions: dict[str, float],
    ) -> list[str]:
        strongest = sorted(feature_contributions.items(), key=lambda item: item[1], reverse=True)[:3]
        lines = []
        if matched_required:
            lines.append(f"Strong required-skill coverage: {', '.join(matched_required[:5])}.")
        if matched_preferred:
            lines.append(f"Preferred skill evidence: {', '.join(matched_preferred[:4])}.")
        if missing_required:
            lines.append(f"Primary gaps: {', '.join(missing_required[:4])}.")
        if candidate.achievements:
            quantified = [item for item in candidate.achievements if any(char.isdigit() for char in item)]
            if quantified:
                lines.append(f"Quantified achievements support seniority: {quantified[0]}")
        lines.append(
            "Top weighted drivers: "
            + ", ".join(f"{feature}={value:.2f}" for feature, value in strongest)
            + "."
        )
        return lines[:6]
