"""Load catalogue additions and define a continuous full-month modeling window."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)
CONTENT_TYPES = ("Movie", "TV Show")


def load_raw(cfg: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Parse dates, normalize placeholders, and exclude rows with unusable dates."""
    raw_path = project_root / cfg["paths"]["raw_data"]
    if not raw_path.is_file():
        raise FileNotFoundError(f"Dataset.csv not found at '{raw_path}'.")
    try:
        df = pd.read_csv(raw_path, dtype=str)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset is empty: {raw_path}") from exc

    date_column = cfg["data"].get("date_column", "date_added")
    missing = {date_column, "type"}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}.")
    if date_column != "date_added":
        df = df.rename(columns={date_column: "date_added"})
    for col in df.columns:
        df[col] = df[col].str.strip()
    if df.empty:
        raise ValueError("Dataset contains no rows.")

    df["date_added"] = pd.to_datetime(df["date_added"], format=cfg["data"]["date_format"], errors="coerce")
    invalid = df["date_added"].isna().sum()
    df = df.dropna(subset=["date_added"]).copy()
    if df.empty:
        raise ValueError("Dataset contains no valid date_added values.")
    if invalid:
        logger.warning("Dropped %d rows with invalid or missing date_added.", invalid)
    if not df["type"].isin(CONTENT_TYPES).all():
        raise ValueError("Content type must be Movie or TV Show.")
    placeholder = cfg["data"].get("placeholder_string", "Not Given")
    for col in ("director", "country"):
        if col in df.columns:
            df[col] = df[col].mask(df[col] == placeholder)
    logger.info("Loaded %d dated catalogue records from %s", len(df), raw_path)
    return df


def build_monthly_series(
    df: pd.DataFrame | dict[str, Any],
    cfg: dict[str, Any] | pd.DataFrame | None = None,
    content_type: str | None = None,
    start_date: str | None = None,
) -> pd.Series:
    """Count additions within the configured complete-month observation window.

    The production policy rejects months without records. A custom dataset with
    known coverage can explicitly allow zero counts for months without rows.
    """
    if isinstance(df, dict) and isinstance(cfg, pd.DataFrame):
        df, cfg = cfg, df
    if not isinstance(df, pd.DataFrame) or not isinstance(cfg, dict):
        raise ValueError("A DataFrame and configuration dict are required.")
    if content_type is not None and content_type not in CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {content_type}.")
    data = cfg["data"]
    if data.get("reindex_freq", "MS") != "MS":
        raise ValueError("Only month-start frequency (MS) is supported.")
    start = pd.Timestamp(start_date or data["modeling_start_date"])
    end = pd.Timestamp(data.get("modeling_end_date", data["full_date_range"]["end"]))
    if (
        pd.isna(start) or pd.isna(end) or start != start.to_period("M").start_time
        or not end.is_month_end or end < start
    ):
        raise ValueError("Modeling window must run from a month start to a later month end.")
    cutoff = data.get("observation_cutoff")
    if cutoff is not None and end > pd.Timestamp(cutoff):
        raise ValueError("Modeling end exceeds the known observation cutoff; exclude partial months.")
    if not {"date_added", "type"}.issubset(df.columns):
        raise ValueError("Monthly series requires date_added and type columns.")
    dates = df["date_added"]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise ValueError("date_added must contain parsed dates.")
    subset = dates[dates.between(start, end)]
    if content_type is not None:
        subset = subset[df.loc[subset.index, "type"].eq(content_type).to_numpy()]
    monthly = subset.groupby(subset.dt.to_period("M").dt.to_timestamp()).size()
    index = pd.date_range(start, end, freq="MS", name="month")
    missing = index.difference(pd.DatetimeIndex(monthly.index))
    policy = data.get("missing_month_policy", "error")
    if policy not in ("error", "zero"):
        raise ValueError("missing_month_policy must be 'error' or 'zero'.")
    if policy == "error" and not missing.empty:
        months = ", ".join(pd.DatetimeIndex(missing).strftime("%Y-%m"))
        raise ValueError(f"Unknown monthly coverage for {content_type or 'All'}: {months}.")
    if subset.empty:
        raise ValueError(f"No dated records in the modeling window for {content_type or 'All'}.")
    return monthly.reindex(index, fill_value=0).rename("additions")
