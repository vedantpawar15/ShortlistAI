"""Candidate reranking interfaces."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from typing import Protocol

from src.logging_utils import logger

PRIMARY_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_MODEL_CACHE_SUBDIR = "sentence_transformers"


class BaseEstimator(Protocol):
    """Minimal estimator protocol used by the optional reranker."""

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> object:
        ...


class Reranker:
    """Apply a second-stage ranking model over retrieved candidates."""

    def __init__(
        self,
        model: BaseEstimator | None = None,
        model_name: str = PRIMARY_RERANKER_MODEL_NAME,
        models_dir: Path | str = "models",
        device: str | None = None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.device = device
        self.loaded_model_name: str | None = None

    def load(self, local_files_only: bool = False) -> None:
        """Load the configured cross-encoder reranker."""
        if self.model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for cross-encoder reranking") from exc

        cache_dir = self.models_dir / DEFAULT_MODEL_CACHE_SUBDIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = CrossEncoder(
            self.model_name,
            cache_folder=str(cache_dir),
            device=self.device,
            local_files_only=local_files_only,
        )
        self.loaded_model_name = self.model_name
        logger.info("Loaded reranker model {}", self.model_name)

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Fit the reranking model when supervised labels are available."""
        if self.model is None:
            return
        self.model.fit(features, labels)

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        """Score query-document pairs with the cross-encoder or a lexical fallback."""
        if not documents:
            return []
        if self.model is None:
            return [lexical_rerank_score(query, document) for document in documents]

        pairs = [[query, document] for document in documents]
        if hasattr(self.model, "predict"):
            scores = self.model.predict(pairs)
            return [float(score) for score in scores]
        raise RuntimeError("Loaded reranker model does not expose predict()")

    def rerank(
        self,
        features: pd.DataFrame,
        query_text: str | None = None,
        candidate_documents: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return candidates ordered by reranking score."""
        if features.empty:
            return features
        if query_text is not None and candidate_documents is not None and len(candidate_documents) == len(features):
            scored = features.copy()
            scored["rerank_score"] = self.score_pairs(query_text, candidate_documents)
            return scored.sort_values(["rerank_score", "score"], ascending=[False, False]).reset_index(drop=True)
        if self.model is None:
            sort_column = "score" if "score" in features else "semantic_similarity"
            return features.sort_values(sort_column, ascending=False).reset_index(drop=True)

        scored = features.copy()
        model_features = scored.select_dtypes(include="number")
        if hasattr(self.model, "predict_proba"):
            scored["rerank_score"] = self.model.predict_proba(model_features)[:, -1]
        else:
            scored["rerank_score"] = self.model.predict(model_features)
        return scored.sort_values("rerank_score", ascending=False).reset_index(drop=True)


def lexical_rerank_score(query: str, document: str) -> float:
    """Return a cheap deterministic reranking score when the cross-encoder is unavailable."""
    query_text = query.lower().strip()
    document_text = document.lower().strip()
    if not query_text or not document_text:
        return 0.0
    return float(SequenceMatcher(None, query_text, document_text).ratio())

