"""Regressions for calendar coverage, fixed-origin evaluation and reconciliation."""
from __future__ import annotations

import csv
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from main import main
from src.data_loader import build_monthly_series, load_raw
from src.evaluation import compute_metrics, evaluate_origin, forecast_bands, save_metrics, walk_forward_cv
from src.feature_engineering import build_features, monthly_index, train_test_masks
from src.forecaster import ForecastingPipeline, forecast_future, instantiate_model


@pytest.mark.parametrize("name", ["XGBoost", "RandomForest"])
def test_recursive_steps_use_prior_predictions(
    name: str, synthetic_series: pd.Series, minimal_cfg: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    history = synthetic_series.iloc[:36]
    model = instantiate_model(name, minimal_cfg)
    lag_values: list[float] = []
    rolling_values: list[float] = []

    def fake_predict(features: pd.DataFrame) -> pd.Series:
        lag = float(features["lag_1"].iloc[0])
        lag_values.append(lag)
        rolling_values.append(float(features["roll_mean_3"].iloc[0]))
        return pd.Series(lag + 1, index=features.index)

    monkeypatch.setattr(model, "_predict_features", fake_predict)
    forecast = model.forecast(history, 3)
    last = float(history.iloc[-1])
    np.testing.assert_allclose(lag_values, [last, last + 1, last + 2])
    np.testing.assert_allclose(rolling_values, [
        history.iloc[-3:].mean(), (history.iloc[-2:].sum() + last + 1) / 3, last + 1,
    ])
    np.testing.assert_allclose(forecast, [last + 1, last + 2, last + 3])


@pytest.mark.parametrize("name", ["XGBoost", "RandomForest"])
def test_changing_holdout_actuals_changes_scores_only(
    name: str, synthetic_series: pd.Series, minimal_cfg: dict[str, Any]
) -> None:
    history, actual = synthetic_series.iloc[:48], synthetic_series.iloc[48:60]
    first, scores = evaluate_origin(instantiate_model(name, minimal_cfg), history, actual)
    second, changed = evaluate_origin(instantiate_model(name, minimal_cfg), history, actual + 1000)
    pd.testing.assert_series_equal(first, second)
    assert first.index.equals(actual.index)
    assert (first >= 0).all()
    assert changed["RMSE"] != scores["RMSE"]


def test_validation_folds_are_expanding_and_end_before_holdout(
    synthetic_series: pd.Series, minimal_cfg: dict[str, Any]
) -> None:
    train = synthetic_series.iloc[:60]
    cv = walk_forward_cv(lambda: instantiate_model("Naive", minimal_cfg), train, minimal_cfg)
    folds = cv.folds
    assert len(folds) == 3
    assert (folds["origin"] < folds["validation_start"]).all()
    assert folds["origin"].is_monotonic_increasing
    assert folds["train_months"].diff().dropna().eq(1).all()
    assert (folds["validation_end"] < monthly_index(synthetic_series)[60].date().isoformat()).all()
    assert folds["horizon"].tolist() == [12, 12, 12]
    assert cv.errors.groupby("origin").size().tolist() == [12, 12, 12]
    with pytest.raises(ValueError, match="Insufficient history"):
        walk_forward_cv(lambda: instantiate_model("Naive", minimal_cfg), train.iloc[:20], minimal_cfg)


@pytest.mark.parametrize("candidate", ["Naive", "SeasonalNaive", "MovingAverage"])
def test_baselines_have_expected_horizon_and_values(
    candidate: str, synthetic_series: pd.Series, minimal_cfg: dict[str, Any]
) -> None:
    history = synthetic_series.iloc[:36]
    points = forecast_future(candidate, history, minimal_cfg, 15)
    assert points.index.equals(synthetic_series.index[36:51])
    if candidate == "SeasonalNaive":
        expected = np.resize(history.iloc[-12:].to_numpy(), 15)
    else:
        window = 1 if candidate == "Naive" else 3
        expected = np.repeat(history.iloc[-window:].mean(), 15)
    np.testing.assert_allclose(points, expected)


@pytest.mark.parametrize("horizon", [0, -1, True, 2.5])
def test_model_input_rejects_invalid_horizon(
    horizon: Any, synthetic_series: pd.Series, minimal_cfg: dict[str, Any]
) -> None:
    for name in ("Naive", "XGBoost", "RandomForest", "SARIMA", "ETS"):
        with pytest.raises(ValueError, match="Forecast horizon"):
            forecast_future(name, synthetic_series, minimal_cfg, horizon)


@pytest.mark.parametrize("name", ["XGBoost", "RandomForest", "SARIMA", "ETS", "SeasonalNaive"])
def test_insufficient_history_is_an_explicit_error(
    name: str, synthetic_series: pd.Series, minimal_cfg: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="Insufficient history"):
        forecast_future(name, synthetic_series.iloc[:2], minimal_cfg, 3)


@pytest.mark.parametrize("field, value", [("lag_months", [0]), ("lag_months", [-1]), ("rolling_windows", [1])])
def test_invalid_feature_windows(
    field: str, value: list[int], synthetic_series: pd.Series, minimal_cfg: dict[str, Any]
) -> None:
    cfg = deepcopy(minimal_cfg)
    cfg["features"][field] = value
    with pytest.raises(ValueError, match="must be an integer"):
        build_features(synthetic_series, cfg)


def test_monthly_index_and_split_validate_gaps_and_counts(synthetic_series: pd.Series) -> None:
    for series in (synthetic_series.iloc[::2], synthetic_series.drop(synthetic_series.index[3])):
        with pytest.raises(ValueError, match="consecutive month starts"):
            monthly_index(series)
    negative = synthetic_series.copy()
    negative.iloc[0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        monthly_index(negative)
    index = monthly_index(synthetic_series)
    with pytest.raises(ValueError, match="overlap"):
        train_test_masks(index, "2020-06-30", "2020-06-01")
    with pytest.raises(ValueError, match="next month start"):
        train_test_masks(index, "2020-06-30", "2020-08-01")


@pytest.mark.parametrize(
    "data, problem",
    [
        ("", "Dataset is empty"),
        ("type,date_added\n", "Dataset contains no rows"),
        ("type,release_year\nMovie,2020\n", "Missing required CSV columns"),
        ("type,date_added\nMovie,broken\n", "no valid date_added"),
    ],
)
def test_csv_validation(data: str, problem: str, minimal_cfg: dict[str, Any], tmp_path: Path) -> None:
    file = tmp_path / "data.csv"
    file.write_text(data, encoding="utf-8")
    cfg = deepcopy(minimal_cfg)
    cfg["paths"]["raw_data"] = str(file)
    with pytest.raises(ValueError, match=problem):
        load_raw(cfg, tmp_path)


def test_partial_month_excluded_and_unknown_coverage_rejected(
    minimal_cfg: dict[str, Any], tmp_path: Path
) -> None:
    cfg = deepcopy(minimal_cfg)
    cfg["data"].update({
        "modeling_start_date": "2021-07-01",
        "modeling_end_date": "2021-08-31",
        "observation_cutoff": "2021-09-25",
        "missing_month_policy": "error",
    })
    file = tmp_path / "data.csv"
    file.write_text("type,date_added\nMovie,7/1/2021\nMovie,8/21/2021\nMovie,9/25/2021\nMovie,broken\n", encoding="utf-8")
    cfg["paths"]["raw_data"] = str(file)
    df = load_raw(cfg, tmp_path)
    assert len(df) == 3
    series = build_monthly_series(df, cfg)
    assert series.tolist() == [1, 1]
    assert monthly_index(series)[-1] == pd.Timestamp("2021-08-01")
    with pytest.raises(ValueError, match="Unknown monthly coverage"):
        build_monthly_series(df.iloc[[0, 2]], cfg)
    cfg["data"]["modeling_end_date"] = "2021-09-30"
    with pytest.raises(ValueError, match="partial months"):
        build_monthly_series(df, cfg)


def test_metric_alignment_undefined_values_and_strict_json(tmp_path: Path) -> None:
    dates = pd.date_range("2021-01-01", periods=3, freq="MS")
    actual = pd.Series([0, 0, 0], index=dates)
    forecast = pd.Series([0, 1, 0], index=dates)
    with pytest.raises(ValueError, match="date indices"):
        compute_metrics(actual, forecast.iloc[::-1])
    metrics = compute_metrics(actual, forecast)
    assert metrics["MAPE"] is None and metrics["R2"] is None and metrics["MASE"] is None
    assert metrics["sMAPE"] == pytest.approx(200 / 3)
    file = save_metrics({"scores": metrics, "nonfinite": [float("nan"), float("inf"), -float("inf")]}, tmp_path)
    assert json.loads(file.read_text()) == {"scores": metrics, "nonfinite": [None, None, None]}


def test_mase_uses_the_origin_history() -> None:
    dates = pd.date_range("2019-01-01", periods=26, freq="MS")
    history = pd.Series(np.arange(24), index=dates[:24])
    actual = pd.Series([20, 30], index=dates[24:])
    forecast = pd.Series([8, 18], index=dates[24:])
    assert compute_metrics(actual, forecast, history)["MASE"] == 1.0
    assert compute_metrics(actual, forecast, history * 0)["MASE"] is None
    with pytest.raises(ValueError, match="precede"):
        compute_metrics(actual, forecast, actual)


def test_band_calibration_uses_only_available_validation_steps() -> None:
    errors = pd.DataFrame({"step": [1, 2] * 5, "error": np.arange(1, 11)})
    points = pd.Series(1.0, index=pd.date_range("2021-01-01", periods=3, freq="MS"))
    bands = forecast_bands(errors, points)
    assert bands["calibration_n"].tolist() == [10, 10, 0]
    assert (bands["lower_95"].iloc[:2] == 0).all()
    assert bands["upper_95"].iloc[2:].isna().all()


def test_batch_reconciles_every_month_and_keeps_direct_forecast(
    synthetic_csv: Path, minimal_cfg: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = deepcopy(minimal_cfg)
    cfg["paths"]["raw_data"] = str(synthetic_csv)
    cfg["data"]["modeling_end_date"] = "2021-08-31"
    cfg["data"]["observation_cutoff"] = "2021-09-25"
    for plot in (
        "plot_feature_importance", "plot_forecast_vs_actual", "plot_future_forecast",
        "plot_historical_patterns_eda", "plot_metrics_comparison", "plot_monthly_trend", "plot_residuals",
    ):
        monkeypatch.setattr(f"src.forecaster.{plot}", lambda *args, **kwargs: None)

    pipeline = ForecastingPipeline(cfg, tmp_path)
    results = pipeline.run_all()
    file, _ = pipeline.save_future_forecasts_and_insights(results)
    with file.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    grouped = {
        label: {row["month"]: Decimal(row["forecast"]) for row in rows if row["content_type"] == label}
        for label in results
    }
    assert len(rows) == 36
    assert len(grouped["All"]) == 12
    assert grouped["All"] == {month: value + grouped["TV Show"][month] for month, value in grouped["Movie"].items()}
    assert all(value.is_finite() and value >= 0 for series in grouped.values() for value in series.values())
    assert (pipeline.metrics_dir / "future_forecast_direct_all.csv").exists()
    assert results["All"]["projection_model"] == "BottomUp"
    assert "reconciled_holdout" in results["All"]


def test_cli_and_notebook_export_the_same_forecasts(
    synthetic_csv: Path, minimal_cfg: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    display_module = pytest.importorskip("IPython.display")
    monkeypatch.setattr(display_module, "display", lambda *args: None)
    monkeypatch.setattr(display_module, "Image", lambda **kwargs: None)
    for plot in (
        "plot_feature_importance", "plot_forecast_vs_actual", "plot_future_forecast",
        "plot_historical_patterns_eda", "plot_metrics_comparison", "plot_monthly_trend", "plot_residuals",
    ):
        monkeypatch.setattr(f"src.forecaster.{plot}", lambda *args, **kwargs: None)

    cfg = deepcopy(minimal_cfg)
    cfg["paths"]["raw_data"] = str(synthetic_csv)
    cfg["paths"]["outputs"] = {name: str(tmp_path / "cli" / name) for name in ("figures", "metrics", "reports")}
    cfg["data"]["modeling_end_date"] = "2021-08-31"
    path = tmp_path / "parity.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    main(["--config", str(path)])

    notebook_output = tmp_path / "notebook"
    monkeypatch.setenv("TASK5_NOTEBOOK_CONFIG", str(path))
    monkeypatch.setenv("TASK5_NOTEBOOK_OUTPUT_DIR", str(notebook_output))
    notebook = json.loads(Path("Netflix_Trend_Forecasting.ipynb").read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "Netflix_Trend_Forecasting.ipynb", "exec"), namespace)

    for name in ("future_forecasts.csv", "evaluation_summary.json"):
        assert (tmp_path / "cli" / "metrics" / name).read_bytes() == (notebook_output / "metrics" / name).read_bytes()
