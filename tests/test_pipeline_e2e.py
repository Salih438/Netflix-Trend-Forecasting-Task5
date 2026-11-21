"""End-to-end integration tests using a synthetic CSV (no real Dataset.csv needed).

All tests use tmp_path fixtures so they are hermetic and reproducible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import load_config
from src.data_loader import build_monthly_series, load_raw
from src.evaluation import compute_metrics
from src.feature_engineering import build_features, time_series_train_test_split
from src.forecaster import ForecastingPipeline
from src.models.ml_forecaster import RandomForestForecaster, XGBoostForecaster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(path: Path, overrides: dict | None = None) -> Path:
    """Write a minimal config.yaml to path and return the file path."""
    cfg = {
        "project": {"name": "test", "version": "1.0.0", "seed": 42},
        "paths": {
            "raw_data": "data/raw/Dataset.csv",
            "processed_data": "data/processed/",
            "outputs": {
                "figures": "outputs/figures/",
                "metrics": "outputs/metrics/",
                "reports": "outputs/reports/",
            },
        },
        "data": {
            "date_column": "date_added",
            "date_format": "%m/%d/%Y",
            "placeholder_string": "Not Given",
            "content_types": ["Movie", "TV Show"],
            "full_date_range": {"start": "2015-01-01", "end": "2021-09-30"},
            "modeling_start_date": "2015-01-01",
            "reindex_freq": "MS",
            "missing_month_policy": "zero",
            "observation_cutoff": "2021-09-25",
            "modeling_end_date": "2021-08-31",
        },
        "features": {
            "lag_months": [1, 2, 3, 6, 12],
            "rolling_windows": [3, 6, 12],
        },
        "models": {
            "future_horizon_months": 12,
            "train_end_date": "2020-06-30",
            "test_start_date": "2020-07-01",
            "statistical": {
                "sarima": {"order": [1, 1, 1], "seasonal_order": [0, 0, 0, 0], "trend": "n"},
                "ets": {
                    "error": "add",
                    "trend": "add",
                    "seasonal": None,
                    "seasonal_periods": 12,
                    "damped_trend": False,
                },
            },
            "ml": {
                "xgboost": {
                    "n_estimators": 5,
                    "max_depth": 2,
                    "learning_rate": 0.1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "random_state": 42,
                },
                "random_forest": {
                    "n_estimators": 5,
                    "max_depth": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                },
            },
        },
        "evaluation": {
            "metrics": ["MAE", "RMSE", "MAPE", "R2"],
            "cross_validation": {"n_splits": 2, "min_train_size": 12},
        },
        "logging": {"level": "WARNING"},
    }
    if overrides:
        cfg.update(overrides)
    cfg_path = path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return cfg_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_load_raw_raises_if_missing(self, tmp_path: Path) -> None:
        cfg_path = _write_config(tmp_path)
        cfg = load_config(cfg_path)
        # Point project root to tmp_path where there's no Dataset.csv
        with pytest.raises(FileNotFoundError, match="Dataset.csv"):
            load_raw(cfg, tmp_path)

    def test_load_raw_cleans_placeholders(self, tmp_path: Path, synthetic_csv: Path) -> None:
        cfg_path = _write_config(tmp_path)
        cfg = load_config(cfg_path)
        # Put CSV in expected location
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "Dataset.csv").write_bytes(synthetic_csv.read_bytes())
        df = load_raw(cfg, tmp_path)
        # "Not Given" should be replaced with NA
        assert (df["director"] == "Not Given").sum() == 0
        assert (df["country"] == "Not Given").sum() == 0

    def test_monthly_series_is_gap_free(self, tmp_path: Path, synthetic_csv: Path) -> None:
        cfg_path = _write_config(tmp_path)
        cfg = load_config(cfg_path)
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "Dataset.csv").write_bytes(synthetic_csv.read_bytes())
        df = load_raw(cfg, tmp_path)
        series = build_monthly_series(df, cfg)
        # The series must be contiguous MS — no gaps
        expected_len = len(pd.date_range("2015-01-01", "2021-08-31", freq="MS"))
        assert len(series) == expected_len, (
            f"Monthly series has {len(series)} rows; expected {expected_len} (gap-free)."
        )


class TestMLForecasters:
    def test_xgboost_predict_shape(self, synthetic_series: pd.Series, minimal_cfg: dict[str, Any]) -> None:
        X, y = build_features(synthetic_series, minimal_cfg)
        X_tr, X_te, y_tr, y_te = time_series_train_test_split(
            X, y, train_end="2020-06-30", test_start="2020-07-01"
        )
        model = XGBoostForecaster(minimal_cfg)
        model.fit(y_tr, X_tr)
        preds = model.predict(len(y_te), X_te)
        assert len(preds) == len(y_te)

    def test_random_forest_predict_non_negative(
        self, synthetic_series: pd.Series, minimal_cfg: dict[str, Any]
    ) -> None:
        X, y = build_features(synthetic_series, minimal_cfg)
        X_tr, X_te, y_tr, y_te = time_series_train_test_split(
            X, y, train_end="2020-06-30", test_start="2020-07-01"
        )
        model = RandomForestForecaster(minimal_cfg)
        model.fit(y_tr, X_tr)
        preds = model.predict(len(y_te), X_te)
        assert (preds >= 0).all(), "Predictions must be non-negative."

    def test_model_raises_before_fit(self, minimal_cfg: dict[str, Any]) -> None:
        model = XGBoostForecaster(minimal_cfg)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.predict(5)


class TestMetrics:
    def test_compute_metrics_perfect_forecast(self) -> None:
        idx = pd.date_range("2021-01-01", periods=6, freq="MS")
        y = pd.Series([10, 20, 30, 40, 50, 60], index=idx, dtype=float)
        metrics = compute_metrics(y, y)
        assert metrics["MAE"] == 0.0
        assert metrics["RMSE"] == 0.0
        assert metrics["R2"] == 1.0

    def test_compute_metrics_handles_zero_actuals(self) -> None:
        idx = pd.date_range("2021-01-01", periods=4, freq="MS")
        y_true = pd.Series([0, 0, 10, 20], index=idx, dtype=float)
        y_pred = pd.Series([1, 1, 11, 21], index=idx, dtype=float)
        metrics = compute_metrics(y_true, y_pred)
        # MAPE is computed only over non-zero actuals → should not be NaN entirely
        assert metrics["MAPE"] is not None and not np.isnan(metrics["MAPE"])


class TestForecastingPipelineE2E:
    def test_pipeline_runs_end_to_end_on_synthetic_data(
        self, tmp_path: Path, synthetic_csv: Path
    ) -> None:
        cfg_path = _write_config(tmp_path)
        cfg = load_config(cfg_path)
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "Dataset.csv").write_bytes(synthetic_csv.read_bytes())

        pipeline = ForecastingPipeline(cfg, tmp_path)
        results = pipeline.run(content_type="Movie")

        assert "metrics" in results
        assert "comparison" in results
        assert "forecasts" in results
        assert "SARIMA" in results["metrics"]
        assert "ETS" in results["metrics"]
        assert "XGBoost" in results["metrics"]
        assert "RandomForest" in results["metrics"]

        # Ensure metrics and figures were written to disk
        assert (tmp_path / "outputs" / "metrics" / "metrics_movie.json").exists()
        assert (tmp_path / "outputs" / "figures" / "forecast_vs_actual_movie.png").exists()
        assert (tmp_path / "outputs" / "figures" / "metrics_comparison_movie.png").exists()

        # Check residual plots with type suffix
        assert (tmp_path / "outputs" / "figures" / "residuals_sarima_movie.png").exists()
        assert (tmp_path / "outputs" / "figures" / "residuals_ets_movie.png").exists()
        assert (tmp_path / "outputs" / "figures" / "residuals_xgboost_movie.png").exists()
        assert (tmp_path / "outputs" / "figures" / "residuals_randomforest_movie.png").exists()

        # Check Step 4 Future Forecast
        assert "future_forecast" in results
        assert len(results["future_forecast"]) == 12
        assert (results["future_forecast"]["forecast"] >= 0).all()
        assert (tmp_path / "outputs" / "figures" / "future_forecast_12m_movie.png").exists()

        # Check EDA historical overview outputs
        assert (tmp_path / "outputs" / "metrics" / "eda_summary.json").exists()
        assert (tmp_path / "outputs" / "figures" / "historical_patterns_eda.png").exists()

        # Test executive reporting and future forecast CSV generation
        csv_path, report_path = pipeline.save_future_forecasts_and_insights({"Movie": results})
        assert csv_path.exists()
        assert report_path.exists()
        report_text = report_path.read_text(encoding="utf-8")
        assert "2021-09-25" in report_text
        assert "partial month" in report_text.lower()
