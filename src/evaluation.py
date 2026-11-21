"""Rolling-origin validation, point scores and strict JSON output."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering import monthly_index, positive_int
from src.models.base import BaseForecaster

Metrics = dict[str, float | None]


def compute_metrics(
    y_true: pd.Series, y_pred: pd.Series, history: pd.Series | None = None
) -> Metrics:
    """Score aligned predictions. Seasonal MASE uses only the origin's history."""
    if y_true.empty or not y_true.index.equals(y_pred.index):
        raise ValueError("Actual and prediction date indices must be nonempty and match exactly.")
    actual, predicted = y_true.to_numpy(dtype=float), y_pred.to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and prediction values must be finite.")

    error = actual - predicted
    absolute_error = np.abs(error)
    mae = float(absolute_error.mean())
    rmse = float(np.sqrt(np.mean(error ** 2)))
    combined = np.abs(actual) + np.abs(predicted)
    smape = float(np.divide(200 * absolute_error, combined, out=np.zeros_like(error), where=combined != 0).mean())
    nonzero = actual != 0
    mape = float(np.mean(100 * absolute_error[nonzero] / np.abs(actual[nonzero]))) if nonzero.any() else None
    centered = actual - actual.mean()
    variance = float(np.sum(centered ** 2))
    r2 = 1.0 - float(np.sum(error ** 2)) / variance if len(actual) > 1 and variance > 0 else None

    mase = None
    if history is not None:
        historical_index = monthly_index(history)
        if not isinstance(y_true.index, pd.DatetimeIndex) or historical_index[-1] >= y_true.index[0]:
            raise ValueError("MASE history must precede the actual evaluation dates.")
        train = history.to_numpy(dtype=float)
        if len(train) > 12:
            scale = float(np.mean(np.abs(train[12:] - train[:-12])))
            if scale > 0.0:
                mase = mae / scale
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "MASE": mase, "R2": r2, "MAPE": mape}


@dataclass(frozen=True)
class BacktestResult:
    folds: pd.DataFrame
    errors: pd.DataFrame

    def mean_metrics(self) -> Metrics:
        result: Metrics = {}
        for name in ("MAE", "RMSE", "sMAPE", "MASE", "R2", "MAPE"):
            score = self.folds[name].mean()
            result[name] = None if pd.isna(score) else float(score)
        return result


def rolling_train_sizes(size: int, cfg: dict[str, Any]) -> tuple[list[int], int]:
    """Build equally spaced, recent expanding-window origins within training data."""
    settings = cfg["evaluation"]["cross_validation"]
    horizon = positive_int(settings.get("horizon_months", cfg["models"]["future_horizon_months"]), "CV horizon")
    step = positive_int(settings.get("step_months", 1), "CV origin step")
    n_splits = positive_int(settings["n_splits"], "CV fold count", 2)
    minimum = positive_int(settings["min_train_size"], "CV minimum history")
    needed = minimum + horizon + step * (n_splits - 1)
    if size < needed:
        raise ValueError(
            f"Insufficient history for rolling-origin validation: need {needed} months, got {size}."
        )
    last = size - horizon
    return [last - step * offset for offset in reversed(range(n_splits))], horizon


def evaluate_origin(
    model: BaseForecaster, history: pd.Series, actual: pd.Series
) -> tuple[pd.Series, Metrics]:
    """Fit and forecast from past values before scoring any future observations."""
    forecast = model.forecast(history, len(actual))
    return forecast, compute_metrics(actual, forecast, history)


def walk_forward_cv(
    model_factory: Callable[[], BaseForecaster], series: pd.Series, cfg: dict[str, Any]
) -> BacktestResult:
    """Use pre-holdout data only and fit a fresh model at each origin."""
    dates = monthly_index(series)
    sizes, horizon = rolling_train_sizes(len(series), cfg)
    folds, errors = [], []
    for fold, size in enumerate(sizes, start=1):
        history = series.iloc[:size]
        actual = series.iloc[size:size + horizon]
        predictions, metrics = evaluate_origin(model_factory(), history, actual)
        origin = dates[size - 1].date().isoformat()
        folds.append({
            "fold": fold,
            "train_start": dates[0].date().isoformat(),
            "origin": origin,
            "train_months": size,
            "validation_start": dates[size].date().isoformat(),
            "validation_end": dates[size + horizon - 1].date().isoformat(),
            "horizon": horizon,
            **metrics,
        })
        errors.append(pd.DataFrame({
            "origin": origin,
            "month": dates[size:size + horizon].strftime("%Y-%m-%d"),
            "step": np.arange(1, horizon + 1),
            "actual": actual.to_numpy(dtype=float),
            "forecast": predictions.to_numpy(dtype=float),
            "error": (actual - predictions).to_numpy(dtype=float),
        }))
    return BacktestResult(pd.DataFrame(folds), pd.concat(errors, ignore_index=True))


def forecast_bands(errors: pd.DataFrame, forecast: pd.Series) -> pd.DataFrame:
    """Use absolute validation errors pooled in four-step groups as exploratory bands."""
    steps = np.arange(1, len(forecast) + 1)
    evaluated_step = int(errors["step"].max())
    groups = (errors["step"] - 1) // 4
    absolute = errors["error"].abs()
    bounds = absolute.groupby(groups).quantile(np.array([0.80, 0.95]), interpolation="higher").unstack()
    lookups = bounds.reindex((steps - 1) // 4)
    sizes = absolute.groupby(groups).size().reindex((steps - 1) // 4, fill_value=0).to_numpy(dtype=int)
    supported = (steps <= evaluated_step) & (sizes >= 5)
    widths_80 = np.where(supported, lookups[0.80].to_numpy(dtype=float), np.nan)
    widths_95 = np.where(supported, lookups[0.95].to_numpy(dtype=float), np.nan)
    points = forecast.to_numpy(dtype=float)
    return pd.DataFrame({
        "forecast": points,
        "lower_80": np.maximum(0.0, points - widths_80),
        "upper_80": points + widths_80,
        "lower_95": np.maximum(0.0, points - widths_95),
        "upper_95": points + widths_95,
        "calibration_n": np.where(steps <= evaluated_step, sizes, 0),
    }, index=forecast.index)


def save_metrics(metrics: dict[str, Any], output_dir: Path, filename: str = "metrics.json") -> Path:
    """Write valid JSON. Undefined numeric results are serialized as null."""
    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value

    path = output_dir / filename
    payload = json.dumps(sanitize(metrics), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def compare_models(results: dict[str, Metrics]) -> pd.DataFrame:
    """Rank by mean validation RMSE, then MAE, then name."""
    frame = pd.DataFrame.from_dict(results, orient="index").rename_axis("model")
    return frame.sort_values(["RMSE", "MAE", "model"])
