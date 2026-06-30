"""High-level orchestration for the candidate ranking pipeline."""

from __future__ import annotations

import pandas as pd

from src.explainer import RankingExplainer
from src.feature_engineering import FeatureEngineer
from src.reranker import Reranker
from src.scoring import CandidateScorer


class RankingEngine:
    """Coordinate retrieval, feature engineering, scoring, and explanations."""

    def __init__(
        self,
        feature_engineer: FeatureEngineer | None = None,
        scorer: CandidateScorer | None = None,
        reranker: Reranker | None = None,
        explainer: RankingExplainer | None = None,
    ) -> None:
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.scorer = scorer or CandidateScorer()
        self.reranker = reranker or Reranker()
        self.explainer = explainer or RankingExplainer()

    def rank_candidates(self, candidates: pd.DataFrame, job: pd.Series) -> pd.DataFrame:
        """Rank candidates for a given job posting."""
        raise NotImplementedError("Ranking orchestration will be implemented later.")

