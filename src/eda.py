"""Aggregate exploration of catalogue addition dates and titles' release years."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return aggregate counts and the observed calendar range."""
    dates = df["date_added"].dropna()
    periods = pd.PeriodIndex(dates.dt.to_period("M"))
    record_months = periods.unique()
    all_months = pd.period_range(record_months.min(), record_months.max(), freq="M")
    early = all_months[all_months.year < 2015]
    summary: dict[str, Any] = {
        "shape": list(df.shape),
        "missing": df.isna().sum().to_dict(),
        "content_type_counts": df["type"].value_counts().to_dict(),
        "date_range": {
            "min": str(dates.min().date()), "max": str(dates.max().date()),
            "n_unique_months": len(record_months), "n_calendar_months": len(all_months),
            "pre_2015_observed_months": len(record_months.intersection(early)),
            "pre_2015_calendar_months": len(early),
        },
    }
    if "country" in df:
        summary["top_countries"] = df["country"].value_counts().head(15).to_dict()
    if "listed_in" in df:
        summary["top_genres"] = (
            df["listed_in"].dropna().str.split(r",\s*").explode().str.strip()
            .value_counts().head(20).to_dict()
        )
    return summary


def compute_eda_summary(
    df: pd.DataFrame, modeling_index: pd.DatetimeIndex | None = None
) -> dict[str, Any]:
    """Compare quarter counts on years with all 12 observed modeling months."""
    summary = summarize_dataset(df)
    dates = df["date_added"]
    years = dates.dt.year
    summary["yearly_growth"] = {str(key): value for key, value in years.value_counts().sort_index().items()}
    counts = pd.DatetimeIndex(dates.dt.to_period("M").dt.to_timestamp().unique())
    if modeling_index is not None:
        counts = counts.intersection(modeling_index)
        summary["modeling_months"] = {
            "start": modeling_index[0].date().isoformat(),
            "end": modeling_index[-1].date().isoformat(),
            "calendar_months": len(modeling_index),
            "observed_months": len(counts),
            "by_type": {
                content_type: len(modeling_index.intersection(pd.DatetimeIndex(
                    group["date_added"].dt.to_period("M").dt.to_timestamp().unique()
                )))
                for content_type, group in df.groupby("type")
            },
        }
    counts_by_year = pd.Series(counts.year).value_counts()
    comparable = sorted(int(str(year)) for year in counts_by_year[counts_by_year == 12].index)
    q_counts = dates[years.isin(comparable)].dt.quarter.value_counts()
    summary["comparable_years"] = comparable
    summary["quarterly_seasonality"] = {
        f"Q{q}": int(q_counts.get(q, 0)) for q in range(1, 5)
    }

    if "release_year" in df:
        release_year = pd.to_numeric(df["release_year"], errors="coerce")
        lag = (years - release_year).to_numpy(dtype=float)
        valid_lag = lag[np.isfinite(lag) & (lag >= 0)]
        if len(valid_lag):
            summary["content_lag_years"] = {
                "mean": float(np.mean(valid_lag)), "median": float(np.median(valid_lag)),
                "std": float(np.std(valid_lag, ddof=1)) if len(valid_lag) > 1 else None,
                "min": int(valid_lag.min()), "max": int(valid_lag.max()),
                "p25": float(np.percentile(valid_lag, 25)), "p75": float(np.percentile(valid_lag, 75)),
                "distribution_brackets": {
                    "same_year_0yr": int((valid_lag == 0).sum()),
                    "1_to_2_years": int(((valid_lag >= 1) & (valid_lag <= 2)).sum()),
                    "3_to_5_years": int(((valid_lag >= 3) & (valid_lag <= 5)).sum()),
                    "6_to_10_years": int(((valid_lag >= 6) & (valid_lag <= 10)).sum()),
                    "over_10_years": int((valid_lag > 10).sum()),
                },
            }
    return summary


def monthly_trend_stats(series: pd.Series) -> dict[str, float]:
    """Return basic numeric summaries for an observed monthly series."""
    stats = series.astype(float)
    return {key: float(stats.agg(key)) for key in ("mean", "std", "min", "max", "median", "skew")}
