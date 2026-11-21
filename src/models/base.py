"""Common interface and fixed-origin forecasting for catalogue count models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering import monthly_index, positive_int


def future_index(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    positive_int(horizon, "Forecast horizon")
    return pd.date_range(index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")


def count_forecast(predictions: pd.Series, expected: pd.DatetimeIndex) -> pd.Series:
    """Check dates and numeric output before clipping count forecasts."""
    if not predictions.index.equals(expected):
        raise ValueError("Forecast dates must match the requested month starts.")
    if not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError("Forecast values must be finite.")
    return predictions.clip(lower=0).rename("forecast")


class BaseForecaster(ABC):
    """Fit and predict with a continuous monthly history."""

    name = "BaseForecaster"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._train_index: pd.DatetimeIndex | None = None

    @abstractmethod
    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> BaseForecaster:
        """Fit on training months only."""

    @abstractmethod
    def predict(self, horizon: int, X_test: pd.DataFrame | None = None) -> pd.Series:
        """Use the fitted model to predict a positive number of months."""

    def _fitted_dates(self, horizon: int) -> pd.DatetimeIndex:
        if self._train_index is None:
            raise RuntimeError("Call fit() before predict().")
        return future_index(self._train_index, horizon)

    def forecast(self, history: pd.Series, horizon: int) -> pd.Series:
        """Forecast the full horizon from one origin using only supplied history."""
        expected = future_index(monthly_index(history), horizon)
        return count_forecast(self.fit(history).predict(horizon), expected)

    def get_params(self) -> dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> BaseForecaster:
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fitted={self._train_index is not None})"
