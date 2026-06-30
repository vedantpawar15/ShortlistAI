"""Candidate reranking interfaces."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator


class Reranker:
    """Apply a second-stage ranking model over retrieved candidates."""

    def __init__(self, model: BaseEstimator | None = None) -> None:
        self.model = model

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Fit the reranking model when supervised labels are available."""
        raise NotImplementedError("Reranker training will be implemented later.")

    def rerank(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return candidates ordered by reranking score."""
        raise NotImplementedError("Reranking will be implemented later.")

