"""Fixed-origin persistence, seasonal-naive and recent-mean baselines."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering import monthly_index, positive_int
from src.models.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    name = "Naive"
    window = 1

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> NaiveForecaster:
        index = monthly_index(y_train)
        if len(index) < self.window:
            raise ValueError(f"Insufficient history for {self.name}: need {self.window} months.")
        self._recent = y_train.iloc[-self.window:].to_numpy(dtype=float)
        self._train_index = index
        return self

    def predict(self, horizon: int, X_test: pd.DataFrame | None = None) -> pd.Series:
        index = self._fitted_dates(horizon)
        return pd.Series(np.resize(self._recent, len(index)), index=index)


class SeasonalNaiveForecaster(NaiveForecaster):
    """Repeat the last 12 observed months for as many steps as requested."""

    name = "SeasonalNaive"
    window = 12


class MovingAverageForecaster(NaiveForecaster):
    """Forecast the same mean of the last training months at every step."""

    name = "MovingAverage"

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.window = positive_int(cfg["models"].get("moving_average_window", 3), "Moving average window")

    def predict(self, horizon: int, X_test: pd.DataFrame | None = None) -> pd.Series:
        index = self._fitted_dates(horizon)
        return pd.Series(float(np.mean(self._recent)), index=index)
