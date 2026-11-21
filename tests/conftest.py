"""Shared pytest fixtures — synthetic dataset, config, and monthly series."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
import yaml


# ---------------------------------------------------------------------------
# Minimal configuration fixture (no I/O — config dict only)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def minimal_cfg(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Return a minimal in-memory config dict for unit tests."""
    tmp = tmp_path_factory.mktemp("cfg")
    cfg: dict[str, Any] = {
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
        },
        "features": {
            "lag_months": [1, 2, 3, 6, 12],
            "rolling_windows": [3, 6, 12],
            "calendar_features": ["month", "quarter", "year", "is_q4"],
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
                    "n_estimators": 10,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "random_state": 42,
                },
                "random_forest": {
                    "n_estimators": 10,
                    "max_depth": 3,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                },
            },
        },
        "evaluation": {
            "metrics": ["MAE", "RMSE", "MAPE", "R2"],
            "cross_validation": {"n_splits": 3, "min_train_size": 12},
        },
        "logging": {"level": "WARNING"},
    }
    # Also write a real YAML file so config.load_config can be tested
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    cfg["_test_config_path"] = str(cfg_path)
    return cfg


# ---------------------------------------------------------------------------
# Synthetic monthly series (gap-free, MS frequency, 2015-01 to 2021-09)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_series() -> pd.Series:
    """Deterministic monthly series with a mild upward trend + seasonality."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2015-01-01", "2021-09-01", freq="MS")
    n = len(idx)
    trend = np.linspace(50, 180, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = rng.normal(0, 5, n)
    values = np.maximum(0, trend + seasonal + noise).astype(int)
    return pd.Series(values, index=idx, name="additions")


# ---------------------------------------------------------------------------
# Synthetic raw DataFrame fixture (mimics Dataset.csv schema)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Return a small raw DataFrame with the expected 10-column schema."""
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2015-01-01", "2021-09-01", freq="MS")
    dates = rng.choice(idx, size=n).astype("datetime64[ns]")
    date_strings = [f"{d.month}/{d.day}/{d.year}" for d in pd.to_datetime(dates)]
    types = rng.choice(["Movie", "TV Show"], size=n)
    directors = rng.choice(["Alice Smith", "Not Given", "Bob Jones"], size=n)
    countries = rng.choice(["United States", "Not Given", "India"], size=n)

    return pd.DataFrame({
        "show_id": [f"s{i}" for i in range(n)],
        "type": types,
        "title": [f"Title {i}" for i in range(n)],
        "director": directors,
        "country": countries,
        "date_added": date_strings,
        "release_year": rng.integers(2000, 2022, size=n),
        "rating": rng.choice(["TV-MA", "TV-14", "PG-13", "R"], size=n),
        "duration": [f"{rng.integers(80, 150)} min" for _ in range(n)],
        "listed_in": rng.choice(["Dramas, International Movies", "Comedies", "Thrillers"], size=n),
    })


# ---------------------------------------------------------------------------
# Synthetic CSV file (tmp_path-based — for integration tests)
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_csv(tmp_path: Path, synthetic_df: pd.DataFrame) -> Path:
    """Write synthetic_df to a temporary CSV file and return the path."""
    csv_path = tmp_path / "Dataset.csv"
    synthetic_df.to_csv(csv_path, index=False)
    return csv_path
