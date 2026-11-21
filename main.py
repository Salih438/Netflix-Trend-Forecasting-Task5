#!/usr/bin/env python3
"""CLI for the monthly Netflix catalogue additions pipeline."""
from __future__ import annotations

import argparse

from src.config import get_project_root, load_config
from src.data_loader import CONTENT_TYPES
from src.forecaster import ForecastingPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Netflix monthly catalogue additions forecasting")
    parser.add_argument(
        "--type", dest="content_type", choices=CONTENT_TYPES,
        help="Forecast one category. Omit to generate the reconciled total and both categories.",
    )
    parser.add_argument("--config", help="Path to config.yaml (default: configs/config.yaml).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pipeline = ForecastingPipeline(load_config(args.config), get_project_root())
    results = pipeline.run_all(args.content_type)
    csv_path, report_path = pipeline.save_future_forecasts_and_insights(results)
    print("\nNetflix monthly catalogue additions\n")
    for label, result in results.items():
        selected = result["best_model"]
        validation = result["comparison"].loc[selected, "RMSE"]
        holdout = result["metrics"][selected]["RMSE"]
        print(
            f"{label}: {selected}, validation RMSE={validation:.2f}, "
            f"holdout RMSE={holdout:.2f}, projection={result['projection_model']}"
        )
    print(f"Projections: {csv_path}\nReport: {report_path}")


if __name__ == "__main__":
    main()
