"""Visualization helpers — all plots saved to disk, never displayed interactively."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # non-interactive backend — safe on headless servers

logger = logging.getLogger(__name__)

# --------------- Shared style -----------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE_WIDE = (14, 5)
FIGSIZE_SQUARE = (8, 6)
DPI = 150


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → '%s'", path)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_monthly_trend(
    series: pd.Series,
    output_dir: Path,
    title: str = "Monthly Netflix Additions",
    filename: str = "monthly_trend.png",
) -> Path:
    """Line chart of monthly content additions over time."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(series.index, series.to_numpy(dtype=float), linewidth=1.5, color="#E50914", label="Additions")
    ax.fill_between(series.index, series.to_numpy(dtype=float), alpha=0.15, color="#E50914")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Additions")
    ax.legend()
    fig.tight_layout()
    out = output_dir / filename
    _save(fig, out)
    return out


def plot_forecast_vs_actual(
    y_true: pd.Series,
    forecasts: dict[str, pd.Series],
    output_dir: Path,
    filename: str = "forecast_vs_actual.png",
) -> Path:
    """Overlay actual vs. multiple model forecasts on one chart."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(y_true.index, y_true.to_numpy(dtype=float), label="Actual", color="black", linewidth=2)
    palette = sns.color_palette("tab10", n_colors=len(forecasts))
    for (model_name, preds), color in zip(forecasts.items(), palette):
        ax.plot(preds.index, preds.to_numpy(dtype=float), label=model_name, color=color,
                linewidth=1.5, linestyle="--")
    ax.set_title("Chronological Holdout — Monthly Netflix Catalogue Additions", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Additions")
    ax.legend()
    fig.tight_layout()
    out = output_dir / filename
    _save(fig, out)
    return out


def plot_metrics_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "metrics_comparison.png",
) -> Path:
    """Plot the mean rolling-origin RMSE used for model selection."""
    chart = comparison_df[["RMSE"]].reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.barplot(data=chart, y="model", x="RMSE", color="steelblue", ax=ax)
    ax.set_title("Rolling-origin validation — mean RMSE", fontsize=14, fontweight="bold")
    ax.set_xlabel("RMSE in monthly additions")
    ax.set_ylabel("")
    fig.tight_layout()
    out = output_dir / filename
    _save(fig, out)
    return out


def plot_feature_importance(
    importances: pd.Series,
    output_dir: Path,
    model_name: str = "XGBoost",
    top_n: int = 15,
    filename: str = "feature_importance.png",
) -> Path:
    """Horizontal bar chart of the top-N most important features."""
    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n // 2)))
    top[::-1].plot(kind="barh", ax=ax, color="#E50914")
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    out = output_dir / filename
    _save(fig, out)
    return out


def plot_residuals(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    output_dir: Path,
    filename: str | None = None,
    content_type: str | None = None,
) -> Path:
    """Scatter plot of residuals (actual − predicted) over time."""
    residuals = y_true - y_pred
    if filename:
        fname = filename
    elif content_type:
        type_suffix = content_type.lower().replace(" ", "_")
        fname = f"residuals_{model_name.lower()}_{type_suffix}.png"
    else:
        fname = f"residuals_{model_name.lower()}.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(residuals.index, residuals.to_numpy(dtype=float), color="steelblue", linewidth=1.2)
    axes[0].axhline(0, linestyle="--", color="red", linewidth=1)
    title_suffix = f" ({content_type})" if content_type else ""
    axes[0].set_title(f"{model_name}{title_suffix} — Residuals over Time")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Residual")

    axes[1].hist(residuals.to_numpy(dtype=float), bins=20, color="steelblue", edgecolor="white")
    axes[1].set_title(f"{model_name}{title_suffix} — Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")

    fig.tight_layout()
    out = output_dir / fname
    _save(fig, out)
    return out


def plot_future_forecast(
    historical_series: pd.Series,
    future_df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    content_type: str = "All",
    filename: str | None = None,
) -> Path:
    """Plot historical additions alongside 12-month future forecasts with prediction intervals.

    Args:
        historical_series: Historical active series (DatetimeIndex).
        future_df: DataFrame with DatetimeIndex and columns:
                   ['forecast', 'lower_80', 'upper_80', 'lower_95', 'upper_95'].
        model_name: Name of the selected winning model.
        output_dir: Output directory for the figure.
        content_type: Content type label (e.g. 'All', 'Movie', 'TV Show').
        filename: Optional custom filename. Defaults to 'future_forecast_12m_<type>.png'.

    Returns:
        Path of the saved figure.
    """
    type_suffix = content_type.lower().replace(" ", "_")
    fname = filename or f"future_forecast_12m_{type_suffix}.png"

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Historical additions (display recent history from 2017 onwards for clean visual clarity if available)
    hist_plot = historical_series[historical_series.index >= pd.Timestamp("2017-01-01")]
    if len(hist_plot) < 24:
        hist_plot = historical_series

    ax.plot(
        hist_plot.index,
        hist_plot.to_numpy(dtype=float),
        label="Historical Additions",
        color="#221F1F",
        linewidth=1.8,
        marker="o",
        markersize=3,
    )

    # Connect the last historical point with the first future point
    bridge_dates = [hist_plot.index[-1], future_df.index[0]]
    bridge_vals = [hist_plot.iloc[-1], future_df["forecast"].iloc[0]]
    ax.plot(bridge_dates, bridge_vals, color="#E50914", linestyle=":", linewidth=1.5)

    # 12-Month Future Forecast
    ax.plot(
        future_df.index,
        future_df["forecast"],
        label=f"{len(future_df)}-month projection ({model_name})",
        color="#E50914",
        linewidth=2.2,
        marker="s",
        markersize=4,
    )

    # Prediction intervals
    ax.fill_between(
        future_df.index,
        future_df["lower_95"].to_numpy(dtype=float),
        future_df["upper_95"].to_numpy(dtype=float),
        color="#E50914",
        alpha=0.15,
        label="95% exploratory prediction band",
    )
    ax.fill_between(
        future_df.index,
        future_df["lower_80"].to_numpy(dtype=float),
        future_df["upper_80"].to_numpy(dtype=float),
        color="#E50914",
        alpha=0.25,
        label="80% exploratory prediction band",
    )

    # Demarcation line
    cutoff_date = hist_plot.index[-1]
    ax.axvline(cutoff_date, color="darkgray", linestyle="--", linewidth=1.2, label="Last modeled month")

    ax.set_title(
        f"Netflix catalogue additions projection — {content_type}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Monthly catalogue additions", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    out = output_dir / fname
    _save(fig, out)
    return out


def plot_historical_patterns_eda(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "historical_patterns_eda.png",
    *,
    summary: dict[str, Any] | None = None,
) -> Path:
    """Plot raw annual counts and quarterly counts for comparable full years."""
    from src.eda import compute_eda_summary

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    if summary is None:
        summary = compute_eda_summary(df)
    valid_df = df.dropna(subset=["date_added"]).copy()
    valid_df["year_added"] = valid_df["date_added"].dt.year
    yearly = summary["yearly_growth"]
    sns.barplot(x=list(yearly), y=list(yearly.values()), ax=axes[0, 0], color="#E50914")
    last_year = max(yearly)
    last_date = valid_df["date_added"].max()
    title = f"Observed annual counts (last record {last_date:%Y-%m-%d})"
    axes[0, 0].set_title(title, fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel(f"Year added ({last_year} may be incomplete)")
    axes[0, 0].set_ylabel("Catalogue additions")
    axes[0, 0].tick_params(axis="x", rotation=45)

    quarters = summary["quarterly_seasonality"]
    years = summary["comparable_years"]
    if years:
        sns.barplot(x=list(quarters), y=list(quarters.values()), ax=axes[0, 1], color="steelblue")
        title = f"Quarter totals across {len(years)} comparable years ({years[0]}–{years[-1]})"
    else:
        title = "Quarter totals unavailable: no full observed year"
    axes[0, 1].set_title(title, fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Quarter")
    axes[0, 1].set_ylabel("Catalogue additions")

    # Panel 3: Top 10 Genres
    if "listed_in" in valid_df.columns:
        genres = (
            valid_df["listed_in"]
            .dropna()
            .str.split(r",\s*")
            .explode()
            .str.strip()
            .value_counts()
            .head(10)
        )
        if not genres.empty:
            genres[::-1].plot(kind="barh", ax=axes[1, 0], color="steelblue")
        axes[1, 0].set_title("Top 10 Content Genres", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Count")

    # Panel 4: Content Lag Distribution (year_added - release_year)
    if "release_year" in valid_df.columns:
        rel_year = pd.to_numeric(valid_df["release_year"], errors="coerce")
        lag = (valid_df["year_added"] - rel_year).dropna()
        lag = lag[(lag >= 0) & (lag <= 25)]  # focus on 0-25 yrs
        if not lag.empty:
            sns.histplot(lag.to_numpy(dtype=float), bins=26, ax=axes[1, 1], color="#221F1F", kde=True)
        axes[1, 1].set_title("Content Lag: Release Year to Netflix Addition (Years)", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Lag in Years (date_added.year - release_year)")
        axes[1, 1].set_ylabel("Count")

    fig.suptitle("Netflix catalogue additions — Observed historical patterns", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()

    out = output_dir / filename
    _save(fig, out)
    return out
