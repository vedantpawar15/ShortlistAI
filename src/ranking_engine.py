"""High-level orchestration for the candidate ranking pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.embedding import EmbeddingModel
from src.explainer import RankingExplainer
from src.feature_engineering import FeatureEngineer
from src.logging_utils import logger
from src.preprocessing import TextPreprocessor
from src.reranker import Reranker
from src.scoring import CandidateScorer
from src.vector_store import FaissVectorStore


class RankingEngine:
    """Coordinate retrieval, feature engineering, scoring, and explanations."""

    def __init__(
        self,
        feature_engineer: FeatureEngineer | None = None,
        scorer: CandidateScorer | None = None,
        reranker: Reranker | None = None,
        explainer: RankingExplainer | None = None,
        embedding_model: EmbeddingModel | None = None,
        vector_store: FaissVectorStore | None = None,
        retrieval_top_k: int | None = None,
    ) -> None:
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.scorer = scorer or CandidateScorer()
        self.reranker = reranker or Reranker()
        self.explainer = explainer or RankingExplainer()
        self.preprocessor = TextPreprocessor()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or FaissVectorStore(Path("outputs") / "candidate_index.faiss")
        self.retrieval_top_k = retrieval_top_k

    def rank_candidates(self, candidates: pd.DataFrame, job: pd.Series) -> pd.DataFrame:
        """Rank candidates for a given job posting."""
        if candidates.empty:
            return pd.DataFrame(columns=["candidate_id", "rank", "score", "reasoning"])

        prepared_candidates = candidates.copy()
        if "semantic_document" not in prepared_candidates.columns:
            profiles = self.preprocessor.build_candidate_profiles(prepared_candidates)
            prepared_candidates["semantic_document"] = [profile.document for profile in profiles]

        job_document = self.preprocessor.build_job_profile(job).document
        retrieved_candidates = self._retrieve_candidates(prepared_candidates, job_document)
        features = self.feature_engineer.build_features(retrieved_candidates, job)
        features["score"] = self.scorer.score(features)
        ranked = self.reranker.rerank(
            features,
            query_text=job_document,
            candidate_documents=list(retrieved_candidates["semantic_document"]),
        )
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked["reasoning"] = [
            self.explainer.explain(str(row["candidate_id"]), row.to_dict()).summary
            for _, row in ranked.iterrows()
        ]
        logger.info("Ranked {} candidates", len(ranked))
        return ranked[["candidate_id", "rank", "score", "reasoning"]]

    def _retrieve_candidates(self, candidates: pd.DataFrame, job_document: str) -> pd.DataFrame:
        """Use semantic retrieval when available, else return all prepared candidates."""
        top_k = self.retrieval_top_k or len(candidates)
        try:
            candidate_result = self.embedding_model.embed_candidates(candidates, use_cache=True, persist=True)
            job_result = self.embedding_model.embed_jobs([job_document], use_cache=True, persist=True)
            metadata = candidates[["candidate_id"]].to_dict(orient="records") if "candidate_id" in candidates else None
            self.vector_store.build(
                candidate_result.embeddings,
                item_ids=list(candidates["candidate_id"].astype(str)) if "candidate_id" in candidates else None,
                metadata=metadata,
            )
            matches = self.vector_store.retrieve(job_result.embeddings[0], top_k=top_k)
            candidate_ids = [match.item_id for match in matches]
            if not candidate_ids:
                return candidates
            ranked_candidates = candidates.assign(
                _retrieval_rank=candidates["candidate_id"].astype(str).map(
                    {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
                )
            )
            ranked_candidates = ranked_candidates.dropna(subset=["_retrieval_rank"])
            ranked_candidates = ranked_candidates.sort_values("_retrieval_rank").drop(columns="_retrieval_rank")
            logger.info("Retrieved {} candidates for reranking", len(ranked_candidates))
            return ranked_candidates.reset_index(drop=True)
        except Exception as exc:
            logger.warning("Semantic retrieval unavailable, falling back to full candidate set: {}", exc)
            return candidates.reset_index(drop=True)

