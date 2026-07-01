"""High-level orchestration for the candidate ranking pipeline.

Hybrid scoring model
--------------------
The pipeline implements a two-stage hybrid ranking approach:

1. **Semantic retrieval** (Stage 1)
   Candidates are embedded with BGE-small and indexed in FAISS.  The top-K
   candidates closest to the job embedding are retrieved.  Cosine similarity
   scores from this stage are stored as ``_semantic_score`` on the retrieved
   frame so that ``FeatureEngineer`` can use them as the ``semantic_similarity``
   feature.

2. **Feature-based scoring** (Stage 2)
   ``FeatureEngineer`` builds a multi-signal feature vector per candidate.
   ``CandidateScorer`` applies calibrated weights to produce a composite score.

3. **Cross-encoder reranking** (Stage 3, optional)
   ``Reranker`` refines the ordering using a cross-encoder or
   Reciprocal Rank Fusion over the stage-1 and stage-2 rank lists.

4. **Explainability**
   ``RankingExplainer.explain_batch()`` adds batch-relative percentile context
   to each candidate's explanation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.embedding import EmbeddingModel
from src.explainer import RankingExplainer
from src.feature_engineering import FeatureEngineer, reciprocal_rank_fusion
from src.logging_utils import logger
from src.preprocessing import TextPreprocessor
from src.reranker import Reranker
from src.scoring import CandidateScorer
from src.vector_store import FaissVectorStore


@dataclass
class RankingMetrics:
    """Statistics reported after a ranking run for diagnostics and logging."""

    total_candidates: int
    retrieved_candidates: int
    ranked_candidates: int
    top_score: float
    mean_score: float
    median_score: float
    score_std: float

    def log(self) -> None:
        """Emit metrics to the configured logger."""
        logger.info(
            "RankingMetrics: total={} retrieved={} ranked={} "
            "top={:.4f} mean={:.4f} median={:.4f} std={:.4f}",
            self.total_candidates,
            self.retrieved_candidates,
            self.ranked_candidates,
            self.top_score,
            self.mean_score,
            self.median_score,
            self.score_std,
        )


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
        hybrid_alpha: float = 0.6,
    ) -> None:
        """Initialise the ranking engine.

        Parameters
        ----------
        hybrid_alpha:
            Blending coefficient for hybrid scoring.  Score is computed as
            ``alpha * stage_1_semantic + (1 - alpha) * stage_2_feature``.
            Only applied when embedding-based retrieval is active.
        """
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.scorer = scorer or CandidateScorer()
        self.reranker = reranker or Reranker()
        self.explainer = explainer or RankingExplainer()
        self.preprocessor = TextPreprocessor()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or FaissVectorStore(Path("outputs") / "candidate_index.faiss")
        self.retrieval_top_k = retrieval_top_k
        self.hybrid_alpha = float(hybrid_alpha)

        # Cache: last candidate fingerprint + the corresponding semantic scores.
        self._candidate_fingerprint: str | None = None
        self._semantic_scores: dict[str, float] = {}

    def rank_candidates(self, candidates: pd.DataFrame, job: pd.Series) -> pd.DataFrame:
        """Rank candidates for a given job posting.

        Returns a DataFrame with columns:
        ``candidate_id``, ``rank``, ``score``, ``reasoning``.
        """
        if candidates.empty:
            return pd.DataFrame(columns=["candidate_id", "rank", "score", "reasoning"])

        prepared_candidates = candidates.copy()
        if "semantic_document" not in prepared_candidates.columns:
            profiles = self.preprocessor.build_candidate_profiles(prepared_candidates)
            prepared_candidates["semantic_document"] = [profile.document for profile in profiles]

        job_document = self.preprocessor.build_job_profile(job).document
        retrieved_candidates, semantic_scores = self._retrieve_candidates(prepared_candidates, job_document)

        # Attach semantic cosine scores to the retrieved frame so that
        # FeatureEngineer can use them as the ``semantic_similarity`` feature.
        if semantic_scores:
            retrieved_candidates["_semantic_score"] = retrieved_candidates["candidate_id"].astype(str).map(
                semantic_scores
            )

        features = self.feature_engineer.build_features(retrieved_candidates, job)
        raw_scores = self.scorer.score(features)

        if semantic_scores and self.hybrid_alpha > 0:
            # Blend embedding cosine scores with the feature-based score.
            embedding_scores = features["semantic_similarity"].clip(0.0, 1.0)
            features["score"] = (
                self.hybrid_alpha * embedding_scores
                + (1.0 - self.hybrid_alpha) * raw_scores
            ).clip(0.0, 1.0)
        else:
            features["score"] = raw_scores

        ranked = self.reranker.rerank(
            features,
            query_text=job_document,
            candidate_documents=list(retrieved_candidates["semantic_document"]),
        )

        # Apply RRF to fuse the feature score rank and the rerank score rank.
        if "rerank_score" in ranked.columns:
            feature_rank = list(ranked.sort_values("score", ascending=False)["candidate_id"].astype(str))
            rerank_rank = list(ranked.sort_values("rerank_score", ascending=False)["candidate_id"].astype(str))
            fused = reciprocal_rank_fusion(feature_rank, rerank_rank)
            id_to_rrf = {cid: score for cid, score in fused}
            ranked["_rrf_score"] = ranked["candidate_id"].astype(str).map(id_to_rrf).fillna(0.0)
            ranked = ranked.sort_values("_rrf_score", ascending=False).reset_index(drop=True)
        else:
            ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)

        ranked["rank"] = range(1, len(ranked) + 1)

        # Generate batch-calibrated explanations.
        explanations = self.explainer.explain_batch(ranked)
        ranked["reasoning"] = [exp.summary for exp in explanations]

        # Drop internal columns before returning.
        internal_cols = [c for c in ranked.columns if c.startswith("_")]
        ranked = ranked.drop(columns=internal_cols, errors="ignore")

        # Compute and log quality metrics.
        scores = ranked["score"].astype(float)
        metrics = RankingMetrics(
            total_candidates=len(candidates),
            retrieved_candidates=len(retrieved_candidates),
            ranked_candidates=len(ranked),
            top_score=float(scores.max()),
            mean_score=float(scores.mean()),
            median_score=float(scores.median()),
            score_std=float(scores.std()),
        )
        metrics.log()

        return ranked

    def _retrieve_candidates(
        self,
        candidates: pd.DataFrame,
        job_document: str,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Use semantic retrieval when available; return all candidates otherwise.

        Returns
        -------
        (retrieved_candidates, semantic_scores)
            ``retrieved_candidates`` — subset/reordering of the input frame.
            ``semantic_scores``      — mapping of candidate_id → cosine score.
                                       Empty dict when embeddings are unavailable.
        """
        top_k = self.retrieval_top_k or len(candidates)
        try:
            candidate_result = self.embedding_model.embed_candidates(candidates, use_cache=True, persist=True)
            job_result = self.embedding_model.embed_jobs([job_document], use_cache=True, persist=True)

            fingerprint = _candidate_fingerprint(candidates)
            if fingerprint != self._candidate_fingerprint or not self.vector_store.is_built:
                metadata = candidates[["candidate_id"]].to_dict(orient="records") if "candidate_id" in candidates else None
                self.vector_store.build(
                    candidate_result.embeddings,
                    item_ids=list(candidates["candidate_id"].astype(str)) if "candidate_id" in candidates else None,
                    metadata=metadata,
                )
                self._candidate_fingerprint = fingerprint

            matches = self.vector_store.retrieve(job_result.embeddings[0], top_k=top_k)
            candidate_ids = [match.item_id for match in matches]
            semantic_scores = {match.item_id: float(match.score) for match in matches}

            if not candidate_ids:
                return candidates.reset_index(drop=True), {}

            ranked_candidates = candidates.assign(
                _retrieval_rank=candidates["candidate_id"].astype(str).map(
                    {cid: idx for idx, cid in enumerate(candidate_ids)}
                )
            )
            ranked_candidates = ranked_candidates.dropna(subset=["_retrieval_rank"])
            ranked_candidates = ranked_candidates.sort_values("_retrieval_rank").drop(columns="_retrieval_rank")
            logger.info("Retrieved {} candidates via semantic search", len(ranked_candidates))
            return ranked_candidates.reset_index(drop=True), semantic_scores

        except Exception as exc:
            logger.warning("Semantic retrieval unavailable, falling back to full candidate set: {}", exc)
            return candidates.reset_index(drop=True), {}


def _candidate_fingerprint(candidates: pd.DataFrame) -> str:
    """Compute a stable hash of the candidate_id column for cache invalidation."""
    if "candidate_id" not in candidates.columns:
        return ""
    ids = sorted(candidates["candidate_id"].astype(str).tolist())
    digest = hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()[:16]
    return digest
