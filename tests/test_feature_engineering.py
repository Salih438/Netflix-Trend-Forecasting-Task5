"""Unit tests for src/feature_engineering.py — leakage, shapes, edge cases."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_features, time_series_train_test_split


class TestBuildFeatures:
    """Tests for build_features."""

    def test_output_shapes_are_consistent(self, synthetic_series: pd.Series, minimal_cfg: dict[str, Any]) -> None:
        X, y = build_features(synthetic_series, minimal_cfg)
        assert len(X) == len(y), "X and y must have equal length."
        assert len(X) > 0, "Feature matrix must be non-empty."

    def test_no_nan_in_output(self, synthetic_series: pd.Series, minimal_cfg: dict[str, Any]) -> None:
        X, y = build_features(synthetic_series, minimal_cfg)
        assert not X.isnull().any().any(), "X must not contain NaN after drop_na=True."
        assert not y.isnull().any(), "y must not contain NaN after drop_na=True."

    def test_expected_columns_present(self, synthetic_series: pd.Series, minimal_cfg: dict[str, Any]) -> None:
        X, _ = build_features(synthetic_series, minimal_cfg)
        for lag in minimal_cfg["features"]["lag_months"]:
            assert f"lag_{lag}" in X.columns, f"lag_{lag} column missing."
        assert "month" in X.columns
        assert "quarter" in X.columns
        assert "year" in X.columns
        assert "is_q4" in X.columns

    def test_lag_1_no_look_ahead(self, minimal_cfg: dict[str, Any]) -> None:
        """lag_1[t] must equal series[t-1] — strict past-only guarantee."""
        idx = pd.date_range("2015-01-01", periods=24, freq="MS")
        series = pd.Series(np.arange(24, dtype=float), index=idx, name="additions")
        X, y = build_features(series, minimal_cfg, drop_na=False)
        # At the first valid lag-1 position (index 1), lag_1 == series[0]
        t_pos = 1
        first_valid = idx[t_pos]
        assert X.loc[first_valid, "lag_1"] == series.iloc[t_pos - 1], (
            "lag_1 must reference the immediately preceding value."
        )

    def test_rolling_mean_no_look_ahead(self, minimal_cfg: dict[str, Any]) -> None:
        """Rolling mean at time t must not use series[t] or later."""
        idx = pd.date_range("2015-01-01", periods=36, freq="MS")
        series = pd.Series(np.arange(36, dtype=float), index=idx, name="additions")
        X, _ = build_features(series, minimal_cfg, drop_na=False)
        # At any valid position p, roll_mean_3 == mean of series[p-3:p]
        col = "roll_mean_3"
        for p in range(14, 20):
            ts = X.index[p]
            expected = series.iloc[p - 3: p].mean()
            actual = X.loc[ts, col]
            if not np.isnan(actual):
                assert abs(actual - expected) < 1e-9, (
                    f"Rolling mean at position {p} used future data."
                )

    def test_is_q4_flag_correct(self, minimal_cfg: dict[str, Any]) -> None:
        idx = pd.date_range("2015-01-01", periods=12, freq="MS")
        series = pd.Series(np.ones(12), index=idx, name="additions")
        X, _ = build_features(series, minimal_cfg, drop_na=False)
        index = pd.DatetimeIndex(X.index)
        q4_months = index[index.month.isin([10, 11, 12])]
        non_q4 = index[~index.month.isin([10, 11, 12])]
        assert (X.loc[q4_months, "is_q4"] == 1).all()
        assert (X.loc[non_q4, "is_q4"] == 0).all()


class TestTrainTestSplit:
    """Tests for time_series_train_test_split."""

    def test_no_overlap(self, synthetic_series: pd.Series, minimal_cfg: dict[str, Any]) -> None:
        X, y = build_features(synthetic_series, minimal_cfg)
        X_train, X_test, y_train, y_test = time_series_train_test_split(
            X, y, train_end="2020-06-30", test_start="2020-07-01"
        )
        overlap = X_train.index.intersection(X_test.index)
        assert len(overlap) == 0, "Train and test sets must not overlap."

    def test_chronological_order(self, synthetic_series: pd.Series, minimal_cfg: dict[str, Any]) -> None:
        X, y = build_features(synthetic_series, minimal_cfg)
        X_train, X_test, _, _ = time_series_train_test_split(
            X, y, train_end="2020-06-30", test_start="2020-07-01"
        )
        assert X_train.index.max() < X_test.index.min(), (
            "All training dates must precede all test dates."
        )

    def test_raises_on_empty_train(self, minimal_cfg: dict[str, Any]) -> None:
        idx = pd.date_range("2019-01-01", periods=12, freq="MS")
        X = pd.DataFrame({"f": range(12)}, index=idx)
        y = pd.Series(range(12), index=idx)
        with pytest.raises(ValueError, match="Training set is empty"):
            time_series_train_test_split(X, y, train_end="2010-12-31", test_start="2011-01-01")

    def test_raises_on_empty_test(self, minimal_cfg: dict[str, Any]) -> None:
        idx = pd.date_range("2019-01-01", periods=12, freq="MS")
        X = pd.DataFrame({"f": range(12)}, index=idx)
        y = pd.Series(range(12), index=idx)
        with pytest.raises(ValueError, match="Test set is empty"):
            time_series_train_test_split(X, y, train_end="2025-12-31", test_start="2026-01-01")
