"""Calendar and history features for continuous month-start count series."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def monthly_index(series: pd.Series) -> pd.DatetimeIndex:
    """Validate ordered, unique, gap-free month starts and finite nonnegative counts."""
    index = series.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans:
        raise ValueError("A nonempty monthly DatetimeIndex is required.")
    if index.tz is not None or not index.equals(
        pd.date_range(index[0], periods=len(index), freq="MS")
    ):
        raise ValueError("Expected ordered, consecutive month starts (MS).")
    if not pd.api.types.is_numeric_dtype(series.dtype):
        raise ValueError("Monthly counts must be numeric.")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Monthly counts must be finite and nonnegative.")
    return index


def positive_int(value: Any, name: str, minimum: int = 1) -> int:
    """Read an integer configuration value without accepting booleans or floats."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def feature_windows(cfg: dict[str, Any]) -> tuple[list[int], list[int]]:
    cfg_features = cfg.get("features", {})
    lags = cfg_features.get("lag_months", [1, 2, 3, 6, 12])
    windows = cfg_features.get("rolling_windows", [3, 6, 12])
    for name, values, minimum in (("lag", lags, 1), ("rolling window", windows, 2)):
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Each {name} configuration must be a sequence.")
        for value in values:
            positive_int(value, name, minimum)
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {name} is not supported.")
    return list(lags), list(windows)


def build_features(
    series: pd.Series, cfg: dict[str, Any], *, drop_na: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Build features at t using only the calendar and counts strictly before t."""
    index = monthly_index(series)
    lags, windows = feature_windows(cfg)
    warmup = max([0, *lags, *windows])
    if drop_na and len(series) <= warmup:
        raise ValueError(f"Insufficient history for features: need at least {warmup + 1} months.")

    features = pd.DataFrame(index=index)
    for lag in lags:
        features[f"lag_{lag}"] = series.shift(lag)
    previous = series.shift(1)
    for window in windows:
        rolling = previous.rolling(window)
        features[f"roll_mean_{window}"] = rolling.mean()
        features[f"roll_std_{window}"] = rolling.std()
    features["month"] = index.month
    features["quarter"] = index.quarter
    features["year"] = index.year
    features["is_q4"] = (index.quarter == 4).astype(int)

    if drop_na:
        features = features.iloc[warmup:]
    return features, series.loc[features.index].copy()


def train_test_masks(index: pd.DatetimeIndex, train_end: str, test_start: str) -> tuple[np.ndarray, np.ndarray]:
    """Describe adjacent train and holdout months on a validated calendar."""
    end, start = pd.Timestamp(train_end), pd.Timestamp(test_start)
    if pd.isna(end) or pd.isna(start):
        raise ValueError("Train and test dates must be specified.")
    end_month, start_month = end.to_period("M"), start.to_period("M")
    if start_month <= end_month:
        raise ValueError("Train and test periods overlap or are out of order.")
    if start_month != end_month + 1 or start != start_month.start_time:
        raise ValueError("Test must start at the next month start after training.")
    train_mask, test_mask = index <= end, index >= start
    if not train_mask.any():
        raise ValueError(f"Training set is empty. Check train_end='{train_end}'.")
    if not test_mask.any():
        raise ValueError(f"Test set is empty. Check test_start='{test_start}'.")
    return train_mask, test_mask


def time_series_train_test_split(
    X: pd.DataFrame, y: pd.Series, train_end: str, test_start: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split adjacent training and test periods on an aligned monthly calendar."""
    index = monthly_index(y)
    if not X.index.equals(index):
        raise ValueError("Feature and target indices must match.")
    train_mask, test_mask = train_test_masks(index, train_end, test_start)
    return X.loc[train_mask], X.loc[test_mask], y.loc[train_mask], y.loc[test_mask]
