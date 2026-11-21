"""XGBoost and Random Forest with shared recursive forecasting."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.feature_engineering import build_features, monthly_index
from src.models.base import BaseForecaster, count_forecast, future_index


class _RecursiveForecaster(BaseForecaster):
    model: XGBRegressor | RandomForestRegressor

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> _RecursiveForecaster:
        index = monthly_index(y_train)
        if len(index) < 2:
            raise ValueError(f"Insufficient history for {self.name}: need two usable training rows.")
        if X_train is None or not X_train.index.equals(index):
            raise ValueError(f"{self.name} requires training features aligned with the target.")
        inputs = X_train.to_numpy(dtype=float)
        if inputs.shape[1] == 0 or not np.isfinite(inputs).all():
            raise ValueError(f"{self.name} training features must be nonempty and finite.")
        self.model.fit(inputs, y_train.to_numpy(dtype=float))
        self._feature_names = X_train.columns.copy()
        self._train_index = index
        return self

    def predict(self, horizon: int, X_test: pd.DataFrame | None = None) -> pd.Series:
        expected = self._fitted_dates(horizon)
        if X_test is None or not X_test.index.equals(expected):
            raise ValueError(f"{self.name} requires features for the requested forecast dates.")
        return self._predict_features(X_test)

    def _predict_features(self, features: pd.DataFrame) -> pd.Series:
        if not features.columns.equals(self._feature_names):
            raise ValueError(f"{self.name} prediction features must match the training columns.")
        inputs = features.to_numpy(dtype=float)
        if not np.isfinite(inputs).all():
            raise ValueError(f"{self.name} prediction features must be finite.")
        return pd.Series(np.maximum(0, self.model.predict(inputs)), index=features.index)

    def forecast(self, history: pd.Series, horizon: int) -> pd.Series:
        """Append each prediction before computing the next month's inputs."""
        expected = future_index(monthly_index(history), horizon)
        X_train, y_train = build_features(history, self.cfg)
        self.fit(y_train, X_train)
        working = history.astype(float).reindex(history.index.append(expected), fill_value=0.0)
        for position in range(len(history), len(working)):
            features, _ = build_features(working.iloc[:position + 1], self.cfg, drop_na=False)
            prediction = float(self._predict_features(features.iloc[[-1]]).iloc[0])
            working.iloc[position] = prediction
        return count_forecast(working.iloc[-horizon:], expected)

    def feature_importances(self) -> pd.Series:
        if self._train_index is None:
            raise RuntimeError("Call fit() before requesting feature importances.")
        return pd.Series(
            self.model.feature_importances_, index=self._feature_names, name="importance"
        ).sort_values(ascending=False)

    def get_params(self) -> dict[str, Any]:
        return self.model.get_params()


class XGBoostForecaster(_RecursiveForecaster):
    name = "XGBoost"

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        settings = cfg["models"]["ml"]["xgboost"]
        self.model = XGBRegressor(
            n_estimators=settings.get("n_estimators", 300),
            max_depth=settings.get("max_depth", 4),
            learning_rate=settings.get("learning_rate", 0.05),
            subsample=settings.get("subsample", 0.8),
            colsample_bytree=settings.get("colsample_bytree", 0.8),
            random_state=settings.get("random_state", 42),
            n_jobs=1,
            verbosity=0,
        )


class RandomForestForecaster(_RecursiveForecaster):
    name = "RandomForest"

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        settings = cfg["models"]["ml"]["random_forest"]
        self.model = RandomForestRegressor(
            n_estimators=settings.get("n_estimators", 200),
            max_depth=settings.get("max_depth", 6),
            min_samples_leaf=settings.get("min_samples_leaf", 2),
            random_state=settings.get("random_state", 42),
            n_jobs=1,
        )
