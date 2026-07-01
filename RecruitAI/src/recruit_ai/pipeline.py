"""End-to-end ranking pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict

from .config import DEFAULT_SCORING_CONFIG, ScoringConfig
from .domain import CandidateProfile, JobRequirement, RankingResult
from .scoring import HybridRanker


class RecruitAIRanker:
    def __init__(self, config: ScoringConfig = DEFAULT_SCORING_CONFIG) -> None:
        self.config = config
        self.ranker = HybridRanker(config=config)

    def rank(self, job: JobRequirement, candidates: list[CandidateProfile]) -> list[RankingResult]:
        return self.ranker.rank(job=job, candidates=candidates)

    def rank_from_dicts(self, job_payload: dict, candidate_payloads: list[dict]) -> list[dict]:
        job = JobRequirement(**job_payload)
        candidates = [CandidateProfile(**payload) for payload in candidate_payloads]
        return [result.to_dict() for result in self.rank(job, candidates)]

    @staticmethod
    def to_json(results: list[RankingResult]) -> str:
        return json.dumps([asdict(result) for result in results], indent=2)
