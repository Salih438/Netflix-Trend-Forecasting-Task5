"""Orchestrate validation, frozen holdout scoring and catalogue projections."""
from __future__ import annotations

from collections.abc import Iterator
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import CONTENT_TYPES, build_monthly_series, load_raw
from src.eda import compute_eda_summary
from src.evaluation import (
    BacktestResult, Metrics, compare_models, compute_metrics, evaluate_origin,
    forecast_bands, rolling_train_sizes, save_metrics, walk_forward_cv,
)
from src.feature_engineering import monthly_index, positive_int, train_test_masks
from src.models.base import BaseForecaster
from src.models.baselines import MovingAverageForecaster, NaiveForecaster, SeasonalNaiveForecaster
from src.models.ml_forecaster import RandomForestForecaster, XGBoostForecaster
from src.models.statistical import ETSForecaster, SARIMAForecaster
from src.visualization import (
    plot_feature_importance, plot_forecast_vs_actual, plot_future_forecast,
    plot_historical_patterns_eda, plot_metrics_comparison, plot_monthly_trend, plot_residuals,
)

logger = logging.getLogger(__name__)
MODEL_CLASSES: dict[str, type[BaseForecaster]] = {
    cls.name: cls for cls in (
        SARIMAForecaster, ETSForecaster, XGBoostForecaster, RandomForestForecaster,
        NaiveForecaster, SeasonalNaiveForecaster, MovingAverageForecaster,
    )
}
BASELINES = ("Naive", "SeasonalNaive", "MovingAverage")
DIRECT_ALL = "All (direct)"
BOTTOM_UP = "BottomUp"


def instantiate_model(model_name: str, cfg: dict[str, Any]) -> BaseForecaster:
    """Create a fresh model for a fold or a final fit."""
    try:
        model_class = MODEL_CLASSES[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc
    return model_class(cfg)


def forecast_future(
    model_name: str, series: pd.Series, cfg: dict[str, Any], horizon: int = 12
) -> pd.Series:
    """Project unseen months using only the supplied complete history."""
    return instantiate_model(model_name, cfg).forecast(series, horizon)


def _difference(predictions: pd.Series, actual: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"actual": actual, "forecast": predictions, "error": actual - predictions})


class ForecastingPipeline:
    """Run the same time-series protocol from the CLI and notebook."""

    def __init__(self, cfg: dict[str, Any], project_root: Path) -> None:
        self.cfg = cfg
        self.project_root = project_root
        self.figures_dir = project_root / cfg["paths"]["outputs"]["figures"]
        self.metrics_dir = project_root / cfg["paths"]["outputs"]["metrics"]
        self.reports_dir = project_root / cfg["paths"]["outputs"]["reports"]
        for directory in (self.figures_dir, self.metrics_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._data: pd.DataFrame | None = None
        self._eda: dict[str, Any] | None = None

    def run(self, content_type: str | None = None) -> dict[str, Any]:
        """Select on training folds, then score each frozen candidate on holdout."""
        if self._data is None:
            self._data = load_raw(self.cfg, self.project_root)
        label = content_type or "All"
        suffix = label.lower().replace(" ", "_")
        series = build_monthly_series(self._data, self.cfg, content_type)
        dates = monthly_index(series)
        if self._eda is None:
            self._eda = compute_eda_summary(self._data, dates)
            save_metrics(self._eda, self.metrics_dir, "eda_summary.json")
            plot_historical_patterns_eda(self._data, self.figures_dir, summary=self._eda)

        settings = self.cfg["models"]
        horizon = positive_int(settings["future_horizon_months"], "Forecast horizon")
        train_mask, test_mask = train_test_masks(dates, settings["train_end_date"], settings["test_start_date"])
        history, holdout = series.loc[train_mask], series.loc[test_mask]
        rolling_train_sizes(len(history), self.cfg)

        backtests: dict[str, BacktestResult] = {}
        for model_name in MODEL_CLASSES:
            factory = MODEL_CLASSES[model_name]
            try:
                backtests[model_name] = walk_forward_cv(lambda: factory(self.cfg), history, self.cfg)
            except ValueError as exc:
                raise ValueError(f"{label} / {model_name} backtest: {exc}") from exc
        validation = {name: run.mean_metrics() for name, run in backtests.items()}
        comparison = compare_models(validation)
        best_model = str(comparison.index[0])
        best_baseline = str(comparison.loc[list(BASELINES)].sort_values(["RMSE", "MAE", "model"]).index[0])
        logger.info("%s selected %s on training backtests (baseline %s)", label, best_model, best_baseline)

        forecasts: dict[str, pd.Series] = {}
        metrics: dict[str, Metrics] = {}
        for name in MODEL_CLASSES:
            model = instantiate_model(name, self.cfg)
            forecasts[name], metrics[name] = evaluate_origin(model, history, holdout)
            if isinstance(model, XGBoostForecaster):
                plot_feature_importance(
                    model.feature_importances(), self.figures_dir, model_name="XGBoost",
                    filename=f"feat_importance_xgb_{suffix}.png",
                )
            if name not in BASELINES:
                plot_residuals(
                    holdout, forecasts[name], name, self.figures_dir,
                    filename=f"residuals_{name.lower()}_{suffix}.png", content_type=label,
                )

        errors = backtests[best_model].errors
        future = forecast_bands(errors, forecast_future(best_model, series, self.cfg, horizon))
        selected_holdout = forecast_bands(errors, forecasts[best_model])

        plot_monthly_trend(series, self.figures_dir, filename=f"monthly_trend_{suffix}.png",
                           title=f"Monthly Netflix catalogue additions — {label}")
        plot_forecast_vs_actual(holdout, forecasts, self.figures_dir, filename=f"forecast_vs_actual_{suffix}.png")
        plot_metrics_comparison(comparison, self.figures_dir, filename=f"metrics_comparison_{suffix}.png")
        plot_future_forecast(series, future, best_model, self.figures_dir, content_type=label,
                             filename=f"future_forecast_{horizon}m_{suffix}.png")

        self._save_backtests(suffix, backtests, validation)
        save_metrics(metrics, self.metrics_dir, f"metrics_{suffix}.json")
        holdout_table = pd.DataFrame({"actual": holdout, **forecasts})
        holdout_table.to_csv(self.metrics_dir / f"holdout_predictions_{suffix}.csv", index_label="month")
        return {
            "series": series, "train": history, "holdout": holdout, "metrics": metrics,
            "backtests": backtests, "forecasts": forecasts, "comparison": comparison,
            "best_model": best_model, "best_baseline": best_baseline,
            "projection_model": best_model, "future_forecast": future,
            "direct_future_forecast": future, "selected_holdout_bands": selected_holdout,
            "type_suffix": suffix,
        }

    def _save_backtests(
        self, suffix: str, results: dict[str, BacktestResult], validation: dict[str, Metrics]
    ) -> None:
        folds = [result.folds.assign(model=model) for model, result in results.items()]
        errors = [result.errors.assign(model=model) for model, result in results.items()]
        pd.concat(folds, ignore_index=True).to_csv(self.metrics_dir / f"backtest_folds_{suffix}.csv", index=False)
        pd.concat(errors, ignore_index=True).to_csv(self.metrics_dir / f"backtest_errors_{suffix}.csv", index=False)
        save_metrics(validation, self.metrics_dir, f"backtest_mean_{suffix}.json")

    def run_all(self, content_type: str | None = None) -> dict[str, dict[str, Any]]:
        """Run a requested segment, or reconcile the full three-series batch."""
        if content_type is not None:
            return {content_type: self.run(content_type)}
        results = {"All": self.run(), **{label: self.run(label) for label in CONTENT_TYPES}}
        self._reconcile(results)
        return results

    def _reconcile(self, results: dict[str, dict[str, Any]]) -> None:
        """Sum child predictions and pool paired aggregate errors."""
        total, movie, tv = (results[label] for label in ("All", *CONTENT_TYPES))
        if total["projection_model"] == BOTTOM_UP:
            return
        for field in ("series", "train", "holdout"):
            if not total[field].equals(movie[field].add(tv[field])):
                raise ValueError(f"Total catalogue history must equal both categories in {field}.")
        movie_errors = movie["backtests"][movie["best_model"]].errors
        tv_errors = tv["backtests"][tv["best_model"]].errors
        keys = ["origin", "month", "step"]
        if not movie_errors[keys].equals(tv_errors[keys]):
            raise ValueError("Bottom-up calibration requires paired origins and steps.")
        errors = movie_errors[keys].assign(error=movie_errors["error"] + tv_errors["error"])
        child_points = [result["future_forecast"]["forecast"] for result in (movie, tv)]
        if not child_points[0].index.equals(child_points[1].index):
            raise ValueError("Projection indices must match before reconciliation.")
        total["future_forecast"] = forecast_bands(errors, child_points[0] + child_points[1])
        total["projection_model"] = BOTTOM_UP

        predicted = movie["forecasts"][movie["best_model"]] + tv["forecasts"][tv["best_model"]]
        total["reconciled_holdout"] = compute_metrics(total["holdout"], predicted, total["train"])
        _difference(predicted, total["holdout"]).to_csv(
            self.metrics_dir / "holdout_reconciled_all.csv", index_label="month"
        )
        total["reconciled_holdout_bands"] = forecast_bands(errors, predicted)

        plot_future_forecast(
            total["series"], total["direct_future_forecast"], total["best_model"], self.figures_dir,
            content_type=DIRECT_ALL, filename="future_forecast_direct_all.png",
        )
        plot_future_forecast(
            total["series"], total["future_forecast"], BOTTOM_UP, self.figures_dir, content_type="All",
            filename=f"future_forecast_{len(total['future_forecast'])}m_all.png",
        )

    def save_future_forecasts_and_insights(
        self, results: dict[str, dict[str, Any]]
    ) -> tuple[Path, Path]:
        """Export projections, run metadata and the catalogue additions report."""
        if not results:
            raise ValueError("Results are required for export.")
        coherent = all(label in results for label in ("All", *CONTENT_TYPES))
        if coherent:
            self._reconcile(results)
        table = pd.DataFrame(self._forecast_rows(results, "future_forecast"))
        if coherent:
            children = table[table["content_type"].isin(CONTENT_TYPES)]
            summed = children.groupby("month")["forecast_cents"].sum()
            is_total = table["content_type"] == "All"
            table.loc[is_total, "forecast_cents"] = table.loc[is_total, "month"].map(summed)
            direct = pd.DataFrame(self._forecast_rows({DIRECT_ALL: results["All"]}, "direct_future_forecast"))
            self._write_projection(direct, self.metrics_dir / "future_forecast_direct_all.csv")
        csv_path = self.metrics_dir / "future_forecasts.csv"
        self._write_projection(table, csv_path)

        summary = self._run_summary(results)
        save_metrics(summary, self.metrics_dir, "evaluation_summary.json")
        report_path = self.reports_dir / "insights.md"
        report_path.write_text(self._executive_report(results, summary, table), encoding="utf-8")
        return csv_path, report_path

    @staticmethod
    def _forecast_rows(
        results: dict[str, dict[str, Any]], field: str
    ) -> Iterator[dict[str, Any]]:
        for label, result in results.items():
            frame = result[field]
            model = result["best_model"] if field == "direct_future_forecast" else result["projection_model"]
            for month, row in frame.iterrows():
                yield {
                    "content_type": label, "month": pd.Timestamp(month).date().isoformat(), "model": model,
                    "forecast_cents": int(np.round(100 * row["forecast"])),
                    **{key: row[key] for key in ("lower_80", "upper_80", "lower_95", "upper_95")},
                    "calibration_n": int(row["calibration_n"]),
                }

    @staticmethod
    def _write_projection(table: pd.DataFrame, path: Path) -> None:
        table = table.copy()
        table["forecast"] = table.pop("forecast_cents") / 100
        # Rounding across the two children can shift the total by one cent.
        for level in (80, 95):
            table[f"lower_{level}"] = np.minimum(table[f"lower_{level}"], table["forecast"])
            table[f"upper_{level}"] = np.maximum(table[f"upper_{level}"], table["forecast"])
        cols = ["content_type", "month", "model", "forecast", "lower_80", "upper_80", "lower_95", "upper_95", "calibration_n"]
        table[cols].to_csv(path, index=False, float_format="%.2f")

    @staticmethod
    def _run_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        first = next(iter(results.values()))
        selected_folds = first["backtests"][first["best_model"]].folds
        chronology = selected_folds[["fold", "train_start", "origin", "validation_start", "validation_end", "horizon"]]
        first_index, last_index = first["train"].index, first["holdout"].index
        forecast_index = first["future_forecast"].index

        def coverage(actual: pd.Series, bands: pd.DataFrame) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for level in (80, 95):
                lower, upper = bands[f"lower_{level}"], bands[f"upper_{level}"]
                eligible = lower.notna() & upper.notna()
                covered = (actual >= lower) & (actual <= upper)
                result[str(level)] = {"hits": covered[eligible].sum(), "months": eligible.sum()}
            return result

        return {
            "target": "Monthly Netflix catalogue additions by date_added",
            "selection_criterion": "Mean rolling-origin RMSE, then mean MAE, then model name",
            "selection_folds": chronology.to_dict(orient="records"),
            "train_start": first_index[0].date().isoformat(),
            "train_origin_month": first_index[-1].date().isoformat(),
            "holdout_start": last_index[0].date().isoformat(),
            "holdout_end": last_index[-1].date().isoformat(),
            "holdout_months": len(last_index),
            "projection_origin_month": last_index[-1].date().isoformat(),
            "projection_start": forecast_index[0].date().isoformat(),
            "projection_end": forecast_index[-1].date().isoformat(),
            "projection_months": len(forecast_index),
            "segments": {
                label: {
                    "selected_model": res["best_model"], "best_validation_baseline": res["best_baseline"],
                    "selected_validation": res["comparison"].loc[res["best_model"]].to_dict(),
                    "baseline_validation": res["comparison"].loc[res["best_baseline"]].to_dict(),
                    "selected_holdout": res["metrics"][res["best_model"]],
                    "baseline_holdout": res["metrics"][res["best_baseline"]],
                    "projection_model": res["projection_model"],
                    "holdout_exploratory_coverage": coverage(res["holdout"], res["selected_holdout_bands"]),
                    **({"bottom_up_holdout": res["reconciled_holdout"],
                        "bottom_up_exploratory_coverage": coverage(res["holdout"], res["reconciled_holdout_bands"])}
                       if "reconciled_holdout" in res else {}),
                }
                for label, res in results.items()
            },
        }

    def _executive_report(
        self, results: dict[str, dict[str, Any]], summary: dict[str, Any], table: pd.DataFrame
    ) -> str:
        settings = self.cfg["data"]
        e = self._eda or {}
        last_record = e.get("date_range", {}).get("max", "unknown")
        coverage_cutoff = settings.get("observation_cutoff", last_record)
        lines = [
            "# Netflix'e eklenen içerik sayısı tahmini",
            "",
            "Hedef, CSV'deki `date_added` üzerinden ay başına katalog eklemesidir. "
            "`release_year` orijinal yayın yılıdır ve tarihsel içerik gecikmesinde kullanılır.",
            f"Son kayıt **{last_record}**, veri kesimi **{coverage_cutoff}**. Tam ay modelleme penceresi "
            f"**{summary['train_start'][:7]}–{summary['projection_origin_month'][:7]}**. "
            "Pencere sonrasındaki kısmi ay (partial month) kayıtları EDA kapsamında görülebilir.",
            "",
            "## Doğrulama ve bağımsız test",
            "",
            f"Tahmin başlangıcı **{summary['train_origin_month'][:7]}**. "
            f"Bağımsız test **{summary['holdout_start'][:7]}–{summary['holdout_end'][:7]}** "
            f"({summary['holdout_months']} tam ay). Model seçimi, aşağıdaki eğitim içi katların ortalama "
            "RMSE değerine göre yapıldı. Eşitlikte MAE ve model adı kullanıldı.",
            "Bir kat boyunca bütün adaylar aynı geçmişe ve aynı aylara erişir. "
            "XGBoost ve Random Forest bir sonraki ayın lag ve rolling özelliklerini kendi tahminleriyle günceller.",
            "",
            "| Kat | Eğitim başlangıcı | Son eğitim ayı | Doğrulama başlangıcı | Doğrulama bitişi | Ufuk |",
            "|---|---|---|---|---|---:|",
        ]
        for fold in summary["selection_folds"]:
            lines.append(
                f"| {fold['fold']} | {fold['train_start'][:7]} | {fold['origin'][:7]} | "
                f"{fold['validation_start'][:7]} | {fold['validation_end'][:7]} | {fold['horizon']} |"
            )
        lines.extend([
            "",
            "| Seri | Seçilen model | CV RMSE | En iyi CV referansı | Referans CV RMSE | Test MAE | Test RMSE | Referans test RMSE |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ])
        for label, result in summary["segments"].items():
            metrics = result["selected_holdout"]
            lines.append(
                f"| {label} | {result['selected_model']} | {result['selected_validation']['RMSE']:.2f} | "
                f"{result['best_validation_baseline']} | {result['baseline_validation']['RMSE']:.2f} | "
                f"{metrics['MAE']:.2f} | {metrics['RMSE']:.2f} | {result['baseline_holdout']['RMSE']:.2f} |"
            )
        lines.extend([
            "",
            "Naive son değeri, SeasonalNaive son 12 ayı ve MovingAverage son eğitim aylarının sabit ortalamasını kullanır.",
            "MASE ölçeği her başlangıçta, o ana kadarki eğitim verisinin 12 aylık mutlak fark ortalamasıdır.",
            "",
            "## Tarihsel projeksiyon ve belirsizlik",
            "",
            f"Tam seriyle yeniden eğitim sonrası **{summary['projection_start'][:7]}–{summary['projection_end'][:7]}** "
            f"için {summary['projection_months']} aylık projeksiyon üretildi. "
            f"Bağımsız test {summary['holdout_months']} adımlık kapsama sahip. Daha uzun adımlarda "
            "dayanak eğitim içi backtest'tir. 2026 için tahmin üretilmedi.",
        ])
        if "All" in results and "reconciled_holdout" in results["All"]:
            coherent_rmse = results["All"]["reconciled_holdout"]["RMSE"]
            lines.append(
                f"Toplam projeksiyon her ay film ve dizi tahminlerinin toplamı. "
                f"Bu toplamın bağımsız test RMSE'si {coherent_rmse:.2f}. "
                "Doğrudan toplam modelinin tahmini ve test skorları ayrı tanı çıktılarında saklanır."
            )
        lines.extend(["", "| Seri | Yayımlanan model | Ufuk toplamı |", "|---|---|---:|"])
        for label, group in table.groupby("content_type", sort=False):
            lines.append(f"| {label} | {group['model'].iloc[0]} | {group['forecast_cents'].sum() / 100:.2f} |")
        lines.extend([
            "",
            "Keşif amaçlı %80 ve %95 bantları eğitim içi doğrulama hatalarının mutlak değerlerine dayanır. "
            "Ardışık dörder adım havuzlanır. Toplam için aynı kat ve adımlardaki alt seri hataları toplanır. "
            "Katlar örtüşebildiği için örnekler bağımlıdır. Yetersiz gözlemli adımların bantları boş bırakılır. "
            "Bağımsız testteki gözlenen kapsama adetleri `evaluation_summary.json` dosyasında yer alır. "
            "Bu kısa test, nominal kapsama düzeylerini doğrulamak için yetersizdir.",
            "",
            "## Veri kapsamı ve tarihsel desen",
            "",
            "Katalog gözlemleri sağlanan CSV ile sınırlıdır. CSV'nin dağıtım lisansı doğrulanmamıştır.",
        ])
        years = e.get("comparable_years", [])
        quarters = e.get("quarterly_seasonality", {})
        if years and quarters:
            lines.append(
                f"Her dört çeyreğin de gözlendiği **{years[0]}–{years[-1]}** tam yılları için "
                + ", ".join(f"{quarter} {value}" for quarter, value in quarters.items())
                + ". Bu sayımlar betimleyicidir."
            )
        return "\n".join(lines) + "\n"
