"""High-level orchestration for the candidate ranking pipeline."""

from __future__ import annotations

import pandas as pd

from src.explainer import RankingExplainer
from src.feature_engineering import FeatureEngineer
from src.logging_utils import logger
from src.preprocessing import TextPreprocessor
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
        self.preprocessor = TextPreprocessor()

    def rank_candidates(self, candidates: pd.DataFrame, job: pd.Series) -> pd.DataFrame:
        """Rank candidates for a given job posting."""
        if candidates.empty:
            return pd.DataFrame(columns=["candidate_id", "rank", "score", "reasoning"])

        prepared_candidates = candidates.copy()
        if "semantic_document" not in prepared_candidates.columns:
            profiles = self.preprocessor.build_candidate_profiles(prepared_candidates)
            prepared_candidates["semantic_document"] = [profile.document for profile in profiles]

        features = self.feature_engineer.build_features(prepared_candidates, job)
        features["score"] = self.scorer.score(features)
        ranked = self.reranker.rerank(features)
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked["reasoning"] = [
            self.explainer.explain(str(row["candidate_id"]), row.to_dict()).summary
            for _, row in ranked.iterrows()
        ]
        logger.info("Ranked {} candidates", len(ranked))
        return ranked[["candidate_id", "rank", "score", "reasoning"]]

