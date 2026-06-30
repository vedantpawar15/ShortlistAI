"""Feature engineering for candidate-job matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer:
    """Create model-ready ranking features from extracted signals."""

    def __init__(self) -> None:
        self.scaler = MinMaxScaler()

    def build_features(self, candidates: pd.DataFrame, job: pd.Series) -> pd.DataFrame:
        """Build ranking features for candidates against one job."""
        raise NotImplementedError("Feature engineering will be implemented later.")

    def normalize_scores(self, values: np.ndarray) -> np.ndarray:
        """Normalize numeric score columns into a comparable range."""
        if values.size == 0:
            return values
        return self.scaler.fit_transform(values.reshape(-1, 1)).ravel()

