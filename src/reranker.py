"""Candidate reranking interfaces."""

from __future__ import annotations

import pandas as pd
from typing import Protocol


class BaseEstimator(Protocol):
    """Minimal estimator protocol used by the optional reranker."""

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> object:
        ...


class Reranker:
    """Apply a second-stage ranking model over retrieved candidates."""

    def __init__(self, model: BaseEstimator | None = None) -> None:
        self.model = model

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Fit the reranking model when supervised labels are available."""
        if self.model is None:
            return
        self.model.fit(features, labels)

    def rerank(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return candidates ordered by reranking score."""
        if features.empty:
            return features
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

