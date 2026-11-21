"""SARIMA and ETS models on the full pre-origin monthly history."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.feature_engineering import monthly_index
from src.models.base import BaseForecaster, count_forecast


class _StatisticalForecaster(BaseForecaster):
    _result: Any

    def predict(self, horizon: int, X_test: pd.DataFrame | None = None) -> pd.Series:
        dates = self._fitted_dates(horizon)
        values = np.asarray(self._result.forecast(horizon), dtype=float)
        return count_forecast(pd.Series(values, index=dates), dates)


class SARIMAForecaster(_StatisticalForecaster):
    name = "SARIMA"

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        settings = cfg["models"]["statistical"]["sarima"]
        self.order: tuple[int, int, int] = settings["order"][0], settings["order"][1], settings["order"][2]
        self.seasonal_order: tuple[int, int, int, int] = tuple(settings["seasonal_order"])
        self.trend: str | None = settings.get("trend", "n")

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> SARIMAForecaster:
        index = monthly_index(y_train)
        p, d, q = self.order
        P, D, Q, period = self.seasonal_order
        minimum = max(3, d + D * period + max(p, q) + max(P, Q) * period + 2)
        if len(index) < minimum:
            raise ValueError(f"Insufficient history for SARIMA: need at least {minimum} months.")
        model = SARIMAX(
            y_train.asfreq("MS"), order=self.order, seasonal_order=self.seasonal_order,
            trend=self.trend, enforce_stationarity=False, enforce_invertibility=False,
        )
        self._result = model.fit(disp=False)
        self._train_index = index
        return self

    def get_params(self) -> dict[str, Any]:
        return {"order": self.order, "seasonal_order": self.seasonal_order, "trend": self.trend}


class ETSForecaster(_StatisticalForecaster):
    name = "ETS"

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        settings = cfg["models"]["statistical"]["ets"]
        self.error: str = settings.get("error", "add")
        self.trend: str | None = settings.get("trend", "add")
        self.seasonal: str | None = settings.get("seasonal", "add")
        self.seasonal_periods: int = settings.get("seasonal_periods", 12)
        self.damped_trend: bool = settings.get("damped_trend", True)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> ETSForecaster:
        index = monthly_index(y_train)
        minimum = 2 * self.seasonal_periods if self.seasonal else 3
        if len(index) < minimum:
            raise ValueError(f"Insufficient history for ETS: need at least {minimum} months.")
        model = ETSModel(
            y_train.asfreq("MS"), error=self.error, trend=self.trend, seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods, damped_trend=self.damped_trend,
        )
        self._result = model.fit(disp=False)
        self._train_index = index
        return self

    def get_params(self) -> dict[str, Any]:
        return {
            "error": self.error,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods,
            "damped_trend": self.damped_trend,
        }
